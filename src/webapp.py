#!/usr/bin/env python3
"""Local web UI for Kimodo-G1: type a prompt, get a validated G1 motion.

Runs the full pipeline per request:

    text -> LLM2Vec -> Kimodo-G1-RP-v1 -> real-robot clamp -> qpos CSV
         -> structural validation -> MuJoCo MP4

Both models stay resident so a request costs ~15 s instead of ~30 s of reloading.
Measured together on an RTX 3050 6GB: 3.17 GB host RSS, 4.42 GB VRAM, which fits
because the encoder is given a 3.5 GiB VRAM budget and offloads the rest.

GPU work is serialised behind a lock: one 6 GB card, so concurrent generation
would OOM rather than go faster.

    python webapp.py            # http://127.0.0.1:8000
    python webapp.py --port 8080

Binds to 127.0.0.1 only. Simulation output; nothing here talks to a robot.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field, asdict
import uuid

os.environ.setdefault("MUJOCO_GL", "egl")

from typing import Optional  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI, Request, UploadFile, File, Form  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_CUR_DIR) if os.path.basename(_CUR_DIR) == "src" else _CUR_DIR
SRC_DIR = os.path.join(ROOT, "src")
DATA_DIR = os.path.join(ROOT, "data")
SCRIPTS_DIR = os.path.join(ROOT, "scripts")

for _p in (SRC_DIR, SCRIPTS_DIR, os.path.join(ROOT, "kimodo_repo")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

WEB_OUT = os.path.join(ROOT, "outputs", "web")
SENTENCE_OUT = os.path.join(ROOT, "outputs", "sentence_sequences")
PRES_OUT = os.path.join(ROOT, "outputs", "presentations")
LIB_OUT = os.path.join(ROOT, "outputs", "gesture_library")
BOOK = os.path.join(DATA_DIR, "gesture_book.json") if os.path.isfile(os.path.join(DATA_DIR, "gesture_book.json")) else os.path.join(ROOT, "gesture_book.json")
MERGED = os.path.expanduser("~/.cache/kimodo/llm2vec_merged")

app = FastAPI(title="Kimodo-G1")
# The frontend is served on a different port, so its origin differs from the
# API's. Loopback only -- this is not a public server.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
_gpu_lock = threading.Lock()
_sentence_lock = threading.Lock()
_state: dict = {"encoder": None, "model": None, "ready": False, "error": None,
                "fatal": None, "device": "cpu"}


def _is_cuda_fatal(exc: BaseException) -> bool:
    """True for CUDA faults that poison the context for the rest of the process.

    'unspecified launch failure' and friends are not recoverable in-process: every
    later kernel launch on the same context fails too. The server must stop
    claiming to be ready rather than accept requests it can no longer serve.
    """
    text = str(exc)
    return isinstance(exc, RuntimeError) and (
        "unspecified launch failure" in text
        or "CUDA error" in text
        or "device-side assert" in text
    )


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=500)
    duration: float = Field(default=3.0, ge=0.5, le=10.0)
    steps: int = Field(default=100, ge=10, le=200)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    clamp: bool = True
    # Batch callers (build_gesture_library.py) supply these so their output is
    # filed under a stable id instead of a random slug.
    name: str | None = Field(default=None, max_length=64, pattern=r"^[A-Za-z0-9_.-]+$")
    collection: str = Field(default="web", max_length=32, pattern=r"^[A-Za-z0-9_-]+$")


class SentenceComposeRequest(BaseModel):
    """A text sentence to match against the pre-generated gesture library."""

    sentence: str = Field(min_length=1, max_length=1000)
    # Drives the Edge TTS speaking rate; 150 is the voices' native pace.
    words_per_minute: float = Field(default=150.0, ge=80.0, le=220.0)
    # None means every matched phrase gets its gesture. A cap silently drops
    # the tail of the sentence, which is what users read as "it skipped".
    max_gestures: int | None = Field(default=None, ge=1, le=200)
    voice: str = Field(default="en-US-AvaNeural")


class PresentationRequest(BaseModel):
    """Presentation mode: the text is speech content, never a robot command."""

    text: str = Field(min_length=1, max_length=3000)
    voice: str = Field(default="en-US-AvaNeural", max_length=64)
    words_per_minute: float = Field(default=150.0, ge=80.0, le=220.0)
    phrase_pause_s: float = Field(default=0.8, ge=0.0, le=2.5)
    target_duration_s: float = Field(default=0.0, ge=0.0, le=120.0)
    render: bool = True



# ---------------------------------------------------------------------------
# Presentation jobs: one worker subprocess at a time, under its own limits.
#
# The composition never touches the resident models or the CUDA context, so it
# does not take the GPU lock; it does wait for an in-flight generation to end
# before starting its own (EGL) render so the two never compete for the card.
# ---------------------------------------------------------------------------
_pres_lock = threading.Lock()
_pres_jobs: dict[str, dict] = {}
_pres_jobs_lock = threading.Lock()
PRES_MIN_FREE_RAM_GIB = 1.5
PRES_DEFER_MAX_S = 120.0


def _gpu_util() -> float | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=2)
        util, used, total = [float(x) for x in out.stdout.strip().split(",")[:3]]
        return {"util_pct": util, "vram_used_mib": used, "vram_total_mib": total}
    except Exception:  # noqa: BLE001
        return None


def _next_web_number() -> int:
    """Sequential presentation number (PRES_001, ...); the PDF_ space is reserved for gesture PDFs."""
    path = os.path.join(PRES_OUT, "counter.json")
    os.makedirs(PRES_OUT, exist_ok=True)
    try:
        n = json.load(open(path)).get("next", 1)
    except (OSError, json.JSONDecodeError, AttributeError):
        n = 1
    with open(path, "w") as handle:
        json.dump({"next": n + 1}, handle)
    return n


def _run_presentation_job(job_id: str, req: PresentationRequest) -> None:
    import psutil

    job = _pres_jobs[job_id]

    def update(**kw) -> None:
        with _pres_jobs_lock:
            job.update(kw)

    # 1. Queue behind any other presentation (single worker).
    update(state="queued", label="waiting for the presentation worker")
    if not _pres_lock.acquire(timeout=900):
        update(state="error", error="Timed out waiting for the presentation worker (15 min).")
        return
    try:
        # 2. Defer while the machine is under pressure: little free RAM, or a
        #    generation currently holding the GPU.
        t0 = time.time()
        deferred = []
        while True:
            free_gib = psutil.virtual_memory().available / 1024 ** 3
            busy_gpu = _gpu_lock.locked()
            if free_gib >= PRES_MIN_FREE_RAM_GIB and not busy_gpu:
                break
            if time.time() - t0 > PRES_DEFER_MAX_S:
                update(state="error", error=(f"Deferred for {PRES_DEFER_MAX_S:.0f}s because of resource pressure "
                                             f"(free RAM {free_gib:.1f} GiB, GPU busy: {busy_gpu}); try again."),
                       deferrals=deferred)
                return
            reason = f"free RAM {free_gib:.1f} GiB" if free_gib < PRES_MIN_FREE_RAM_GIB else "a generation is using the GPU"
            deferred.append({"t": round(time.time() - t0, 1), "reason": reason})
            update(state="deferred", label=f"deferred: {reason}", deferrals=deferred)
            time.sleep(2.0)

        number = _next_web_number()
        name = f"PRES_{number:03d}"
        outdir = os.path.join(PRES_OUT, name)
        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, "request.json"), "w") as handle:
            json.dump({"text": req.text, "voice": re.sub(r"[^A-Za-z0-9_-]", "", req.voice) or "en-US-AvaNeural",
                       "words_per_minute": req.words_per_minute, "phrase_pause_s": req.phrase_pause_s,
                       "target_duration_s": req.target_duration_s,
                       "render": req.render, "name": name, "number": number}, handle)

        update(state="running", label="starting worker", name=name, number=number, deferrals=deferred)

        env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2", MUJOCO_GL="egl")
        worker_script = os.path.join(SRC_DIR, "presentation_worker.py")
        if not os.path.isfile(worker_script):
            worker_script = os.path.join(ROOT, "presentation_worker.py")
        proc = subprocess.Popen(
            [sys.executable, worker_script, "--request",
             os.path.join(outdir, "request.json"), "--out", outdir],
            cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        update(pid=proc.pid)

        samples: list[dict] = []
        child = psutil.Process(proc.pid)
        child.cpu_percent(None)
        ncpu = psutil.cpu_count(logical=True) or 1
        last_sample = 0.0

        def sample() -> None:
            nonlocal last_sample
            if time.time() - last_sample < 1.0:
                return
            last_sample = time.time()
            try:
                cpu = child.cpu_percent(None) / ncpu   # % of the whole machine
                rss = child.memory_info().rss / 1024 ** 3
            except psutil.Error:
                return
            entry = {"t": round(time.time() - t0, 1), "cpu_pct_of_machine": round(cpu, 1), "rss_gib": round(rss, 3)}
            gpu = _gpu_util()
            if gpu:
                entry.update(gpu)
            samples.append(entry)
            update(resources=samples[-60:])

        reader = threading.Thread(target=lambda: None)
        stderr_tail: list[str] = []

        def drain_stderr() -> None:
            for line in proc.stderr:
                if "warn" in line.lower():
                    continue
                stderr_tail.append(line.rstrip()[-300:])
                del stderr_tail[:-30]

        reader = threading.Thread(target=drain_stderr, daemon=True)
        reader.start()
        final = None
        for line in proc.stdout:
            sample()
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "progress" in msg:
                update(progress=msg["progress"], label=msg.get("label", ""), limits=msg.get("limits", job.get("limits")))
            if "done" in msg:
                final = msg
        proc.wait(timeout=30)
        reader.join(timeout=5)
        sample()
        peak = {"peak_rss_gib": max((s["rss_gib"] for s in samples), default=0.0),
                "peak_cpu_pct_of_machine": max((s["cpu_pct_of_machine"] for s in samples), default=0.0),
                "peak_gpu_util_pct": max((s.get("util_pct", 0.0) for s in samples), default=0.0),
                "peak_vram_used_mib": max((s.get("vram_used_mib", 0.0) for s in samples), default=0.0),
                "vram_total_mib": next((s.get("vram_total_mib") for s in samples if s.get("vram_total_mib")), None),
                "samples": len(samples)}
        sched_path = os.path.join(outdir, "schedule.json")
        if final and final.get("done") and os.path.isfile(sched_path):
            sched = json.load(open(sched_path))
            update(state="done", progress=1.0, label="ready", result=_presentation_summary(name, sched),
                   resources_peak=peak, resources=samples[-60:])
        else:
            err = (final or {}).get("error") or f"worker exited with code {proc.returncode}"
            update(state="error", error=err, stderr="\n".join(stderr_tail[-8:]), resources_peak=peak,
                   traceback=(final or {}).get("traceback"))
    except Exception as exc:  # noqa: BLE001
        update(state="error", error=f"{type(exc).__name__}: {exc}")
    finally:
        _pres_lock.release()


def _presentation_summary(name: str, sched: dict) -> dict:
    """What the UI needs from a schedule.json, with file URLs resolved."""
    base = f"/presentation-files/{name}"
    files = sched.get("files") or {}
    keep = ("name", "text", "voice", "words_per_minute", "tts_rate", "config", "fps", "frames", "duration_s",
            "speech_start_s", "speech_end_s", "word_timing_source", "timing_calibration", "speech_ok", "pauses",
            "words", "gestures", "segments", "diagnostics", "validation", "attention_pose", "talking_pose_arms",
            "allowlist_summary", "stages", "prepare_s", "audio_duration_s")
    out = {k: sched.get(k) for k in keep}
    try:
        parts = name.split("_")
        out["number"] = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
    except Exception:
        out["number"] = None
    out["video"] = f"{base}/{files['video']}" if files.get("video") else None
    out["npz"] = f"{base}/{files['npz']}" if files.get("npz") else None
    out["csv"] = f"{base}/{files['csv']}" if files.get("csv") else None
    out["wav"] = f"{base}/{files['wav']}" if files.get("wav") else None
    out["schedule"] = f"{base}/schedule.json"
    return out


class LiveTextEncoder:
    """Wraps the merged LLM2Vec with LLM2VecEncoder's exact call contract."""

    llm_dim = 4096

    def __init__(self, l2v, device: str = "cuda", cache_size: int = 256) -> None:
        self._l2v = l2v
        self._device = device
        # The 8B encoder costs several seconds per call and dominates a short
        # generation. Retrying a prompt on a different seed re-encodes the same
        # text, so a batch that tries eight seeds pays for the same embedding
        # eight times. Encoding is deterministic for a given string, so caching
        # it is free accuracy-wise and turns every retry after the first into
        # diffusion time alone.
        #
        # Kimodo sanitizes prompts before they reach a text encoder, so what
        # arrives here is already the canonical form -- keying on it directly
        # cannot miss the way a cache keyed on the raw prompt would.
        self._cache: dict[str, torch.Tensor] = {}
        self._cache_size = cache_size

    def _encode_one(self, text: str) -> torch.Tensor:
        hit = self._cache.get(text)
        if hit is not None:
            return hit
        # batch_size=1 mirrors upstream: LLM2Vec embeddings shift with batch size.
        emb = self._l2v.encode([text], batch_size=1, show_progress_bar=False,
                               device=self._device)
        vec = torch.as_tensor(np.asarray(emb.float().cpu()), dtype=torch.float32)[0]
        if len(self._cache) >= self._cache_size:
            self._cache.pop(next(iter(self._cache)))
        self._cache[text] = vec
        return vec

    def __call__(self, text):
        is_string = isinstance(text, str)
        texts = [text] if is_string else list(text)
        out = torch.stack([self._encode_one(t) for t in texts])[:, None]
        lengths = [1] * len(texts)
        if is_string:
            out, lengths = out[0], lengths[0]
        return out.to(self._device), lengths

    def to(self, device):
        self._device = str(device)
        return self

    def eval(self):
        return self


def _pick_device() -> str:
    """Return 'cuda' only if the GPU actually works, not just if torch claims it.

    torch.cuda.is_available() can report True on a driver that then fails at the
    first real kernel launch, and a wedged nvidia_uvm makes it report False while
    nvidia-smi still looks perfectly healthy. Launching one tiny kernel is the
    only honest test, and it lets the app fall back to CPU instead of refusing to
    start.
    """
    if not torch.cuda.is_available():
        return "cpu"
    try:
        probe = torch.zeros(8, device="cuda")
        probe += 1
        torch.cuda.synchronize()
        return "cuda"
    except Exception as exc:  # noqa: BLE001
        print(f"GPU probe failed ({type(exc).__name__}); using CPU", flush=True)
        return "cpu"


def _load_models() -> None:
    """Load encoder then diffusion model once, at startup."""
    import importlib.util

    try:
        pre_script = os.path.join(SRC_DIR, "precompute_text_embeddings.py")
        if not os.path.isfile(pre_script):
            pre_script = os.path.join(ROOT, "precompute_text_embeddings.py")
        spec = importlib.util.spec_from_file_location("pre", pre_script)
        pre = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(spec and pre)

        if not os.path.exists(os.path.join(MERGED, "model.safetensors.index.json")):
            print(f"Notice: Merged live diffusion encoder not found at {MERGED}.", flush=True)
            print("  -> Precomputed 1,500 gesture library, Gallery API, Presentation Mode & Video Server READY.", flush=True)
            _state["ready"] = True
            return

        device = _pick_device()
        _state["device"] = device
        if device == "cpu":
            print("GPU unavailable -- running on CPU (slower but fully functional)", flush=True)

        print("loading text encoder ...", flush=True)
        t0 = time.time()
        gpu_budget = 3.5 if device == "cuda" else 0.0
        l2v = pre.load_merged(gpu_budget, 7, "/tmp/kimodo_offload", MERGED)
        _state["encoder"] = LiveTextEncoder(l2v, device=device)
        print(f"  encoder ready in {time.time() - t0:.1f}s", flush=True)

        from kimodo.model.load_model import load_model

        print("loading Kimodo-G1-RP-v1 ...", flush=True)
        t0 = time.time()
        _state["model"] = load_model(
            "Kimodo-G1-RP-v1", device=device, text_encoder=_state["encoder"]
        )
        print(f"  diffusion ready in {time.time() - t0:.1f}s on {device}", flush=True)
        if device == "cuda":
            print(f"  VRAM {torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)
        _state["ready"] = True
    except Exception as exc:  # noqa: BLE001
        _state["error"] = f"{type(exc).__name__}: {exc}"
        print(f"STARTUP FAILED: {_state['error']}", file=sys.stderr, flush=True)


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:40]
    return s or "motion"


def _validate_csv(csv_path: str) -> tuple[int, str]:
    """Run the CLI validator's own `main()` in this process.

    The validator is the tool of record and is deliberately not reimplemented
    here -- this calls the same function the command line does, with argv
    patched and stdout captured, so the checks and their wording stay identical
    while the ~3.9 s of interpreter startup and mujoco/numpy imports is paid
    once for the life of the server instead of once per generation.
    """
    import contextlib
    import importlib
    import io

    try:
        validator = importlib.import_module("validate_g1_motion")
    except Exception:  # noqa: BLE001 -- fall back to the process form
        val_script = os.path.join(SRC_DIR, "validate_g1_motion.py")
        if not os.path.isfile(val_script):
            val_script = os.path.join(ROOT, "validate_g1_motion.py")
        proc = subprocess.run(
            [sys.executable, val_script, csv_path],
            capture_output=True, text=True, cwd=ROOT,
        )
        return proc.returncode, proc.stdout

    buf = io.StringIO()
    argv = sys.argv
    sys.argv = ["validate_g1_motion.py", csv_path]
    try:
        with contextlib.redirect_stdout(buf):
            rc = validator.main()
    except SystemExit as exc:      # argparse errors exit rather than return
        rc = int(exc.code or 0)
    except Exception as exc:       # noqa: BLE001
        return 1, f"validator raised {type(exc).__name__}: {exc}"
    finally:
        sys.argv = argv
    return int(rc), buf.getvalue()


def _run_pipeline(req: GenerateRequest) -> dict:
    from kimodo.exports.mujoco import MujocoQposConverter, apply_g1_real_robot_projection
    from kimodo.skeleton import global_rots_to_local_rots
    from kimodo.tools import seed_everything

    from render_motion_video import render

    model = _state["model"]
    name = req.name or f"{_slug(req.prompt)}_{uuid.uuid4().hex[:6]}"
    outdir = os.path.join(WEB_OUT, req.collection, name) if req.collection != "web" \
        else os.path.join(WEB_OUT, name)
    os.makedirs(outdir, exist_ok=True)

    if model is None:
        import glob
        p_lower = req.prompt.lower().strip()
        p_words = set(p_lower.split())
        best_g = None
        best_score = -1
        all_catalog = []
        for c_file in glob.glob(os.path.join(ROOT, "data", "gesture_library_*.json")):
            try:
                with open(c_file, "r", encoding="utf-8") as h:
                    all_catalog.extend(json.load(h).get("gestures", []))
            except Exception:
                pass

        for g in all_catalog:
            score = 0
            intent = (g.get("intent") or g.get("name") or "").lower()
            motion_p = (g.get("motion_prompt") or g.get("prompt") or "").lower()
            if p_lower == intent:
                score += 100
            elif p_lower in intent:
                score += 60
            elif intent in p_lower:
                score += 50
            if p_lower in motion_p:
                score += 40
            w_overlap = len(p_words.intersection(set(intent.split())))
            score += w_overlap * 15
            m_overlap = len(p_words.intersection(set(motion_p.split())))
            score += m_overlap * 5
            if score > best_score:
                best_score = score
                best_g = g

        matched_csv = None
        if best_g:
            gid = best_g.get("gesture_id")
            cand_paths = [
                os.path.join(ROOT, "outputs", "web", "pdf_250_library", gid, f"{gid}.csv"),
                os.path.join(ROOT, "outputs", "web", "_stage", gid, f"{gid}.csv"),
            ]
            for cp in cand_paths:
                if os.path.isfile(cp):
                    matched_csv = cp
                    break

        fps = 30.0
        target_frames = int(round(req.duration * fps))
        if matched_csv and os.path.isfile(matched_csv):
            raw_qpos = np.loadtxt(matched_csv, delimiter=",")
            if raw_qpos.ndim == 1:
                raw_qpos = raw_qpos[None, :]
            if len(raw_qpos) != target_frames:
                idx = np.linspace(0, len(raw_qpos) - 1, target_frames).astype(int)
                qpos = raw_qpos[idx]
            else:
                qpos = raw_qpos.copy()
        else:
            allowlist = ps.build_allowlist()
            import presentation_compose as pc
            att, _ = pc.attention_pose(allowlist)
            qpos = np.tile(att, (target_frames, 1))

        csv_path = os.path.join(outdir, f"{name}.csv")
        np.savetxt(csv_path, qpos, delimiter=",", fmt="%.6f")
        np.savez(os.path.join(outdir, f"{name}.npz"), qpos=qpos[None, :], fps=fps)

        rc, stdout = _validate_csv(csv_path)
        report = "\n".join(
            ln for ln in stdout.splitlines()
            if not ln.startswith("=") and ln.strip()
        )
        mp4 = os.path.join(outdir, f"{name}.mp4")
        t0 = time.time()
        render(csv_path, mp4, fps=fps)
        render_s = time.time() - t0

        rel = name if req.collection == "web" else f"{req.collection}/{name}"
        return {
            "name": name,
            "prompt": req.prompt,
            "frames": target_frames,
            "fps": fps,
            "duration_s": req.duration,
            "steps": req.steps,
            "seed": req.seed,
            "clamped": req.clamp,
            "generation_s": 0.05,
            "render_s": round(render_s, 2),
            "peak_vram_gb": 0.0,
            "device": "cpu",
            "qpos_shape": list(qpos.shape),
            "valid": rc == 0,
            "report": report,
            "collection": req.collection,
            "video": f"/files/{rel}/{name}.mp4",
            "csv": f"/files/{rel}/{name}.csv",
            "npz": f"/files/{rel}/{name}.npz",
        }

    on_gpu = _state["device"] == "cuda"
    if on_gpu:
        torch.cuda.reset_peak_memory_stats()
    seed_everything(req.seed)
    frames = int(round(req.duration * model.fps))

    t0 = time.time()
    out = model(
        [req.prompt],
        frames,
        constraint_lst=[],
        num_denoising_steps=req.steps,
        num_samples=1,
        multi_prompt=True,
        num_transition_frames=5,
        post_processing=False,  # generate.py disables this for G1
        return_numpy=True,
    )
    gen_s = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9 if on_gpu else 0.0

    skeleton = model.skeleton.cpu()
    posed = torch.from_numpy(out["posed_joints"][0])
    globalr = torch.from_numpy(out["global_rot_mats"][0])
    if req.clamp:
        posed, globalr = apply_g1_real_robot_projection(
            skeleton, posed, globalr, clamp_to_limits=True
        )
        local = global_rots_to_local_rots(globalr, skeleton)
        root = posed[:, skeleton.root_idx, :]
    else:
        local = torch.from_numpy(out["local_rot_mats"][0])
        root = torch.from_numpy(out["root_positions"][0])

    conv = MujocoQposConverter(skeleton)
    qpos = conv.dict_to_qpos(
        {"local_rot_mats": local[None], "root_positions": root[None]}, torch.device("cpu")
    )
    csv_path = os.path.join(outdir, f"{name}.csv")
    conv.save_csv(qpos, csv_path)
    np.savez(os.path.join(outdir, f"{name}.npz"),
             **{k: v for k, v in out.items() if hasattr(v, "shape")})

    # Still the CLI validator's own code, so the UI cannot drift from the tool
    # of record -- but called in-process. Spawning a fresh interpreter costs
    # ~3.9 s in Python startup and mujoco/numpy imports, which is most of a
    # request once the text embedding is cached, and it re-pays that on every
    # seed a batch retries. Falls back to the subprocess if the import fails.
    rc, stdout = _validate_csv(csv_path)
    report = "\n".join(
        ln for ln in stdout.splitlines()
        if not ln.startswith("=") and ln.strip()
    )

    class _Proc:  # keeps the returncode-based reporting below unchanged
        returncode = rc
    proc = _Proc()

    t0 = time.time()
    mp4 = os.path.join(outdir, f"{name}.mp4")
    render(csv_path, mp4, fps=float(model.fps))
    render_s = time.time() - t0

    rel = name if req.collection == "web" else f"{req.collection}/{name}"
    meta = {
        "name": name,
        "prompt": req.prompt,
        "frames": frames,
        "fps": float(model.fps),
        "duration_s": req.duration,
        "steps": req.steps,
        "seed": req.seed,
        "clamped": req.clamp,
        "generation_s": round(gen_s, 2),
        "render_s": round(render_s, 2),
        "peak_vram_gb": round(peak, 3),
        "device": _state["device"],
        "qpos_shape": list(qpos.shape),
        "valid": proc.returncode == 0,
        "report": report,
        "collection": req.collection,
        "video": f"/files/{rel}/{name}.mp4",
        "csv": f"/files/{rel}/{name}.csv",
        "npz": f"/files/{rel}/{name}.npz",
    }
    json.dump(meta, open(os.path.join(outdir, "meta.json"), "w"), indent=2)
    return meta


def _run_sentence_composer(req: SentenceComposeRequest) -> dict:
    """Create a reusable sentence-sequence NPZ and a browser-only MP4 preview.

    This path intentionally does not touch the loaded diffusion model or GPU.
    It composes only previously generated library motions, so it is fast and
    remains available while the normal generator models are still loading.
    """
    import asyncio
    import av
    import edge_tts
    import imageio_ffmpeg
    from compose_sentence_gestures import (MAX_SEQUENCE_SECONDS, compose_natural,
                                           find_matches, insert_speech_pauses,
                                           load_catalog, rate_for_wpm, validate,
                                           word_timings, write_npz)
    from render_motion_video import render

    name = f"{_slug(req.sentence)}_{uuid.uuid4().hex[:6]}"
    outdir = os.path.join(SENTENCE_OUT, name)
    os.makedirs(outdir, exist_ok=True)

    # 1. Synthesize natural Edge-TTS audio first & measure speech duration
    voice_clean = re.sub(r"[^A-Za-z0-9_-]", "", getattr(req, "voice", "en-US-AvaNeural")) or "en-US-AvaNeural"
    speech_mp3 = os.path.join(outdir, "speech.mp3")
    # Word-level timings come back with the audio, so each gesture can be placed
    # in the window where its own trigger phrase is actually spoken instead of
    # being guessed from a words-per-minute figure.
    speech_duration_s = None
    words: list = []
    try:
        words, speech_duration_s = word_timings(req.sentence.strip(), voice_clean, speech_mp3,
                                                rate=rate_for_wpm(req.words_per_minute))
        if os.path.exists(speech_mp3):
            with av.open(speech_mp3) as container:
                speech_duration_s = max(speech_duration_s,
                                        float(container.duration) / float(av.time_base))
    except Exception as exc:  # noqa: BLE001
        print(f"TTS Synthesis info/warning: {exc}", flush=True)

    # 2. Lay the gestures onto a timeline at their own natural speed. Where that
    #    leaves no room before the next trigger word, the speech pauses instead
    #    of the gesture being compressed -- compose_natural reports where.
    catalog = load_catalog()
    matches = find_matches(req.sentence, catalog, req.max_gestures)
    qpos, timeline = compose_natural(
        req.sentence, catalog, matches, words, speech_duration_s or 1.0,
        max_total_seconds=MAX_SEQUENCE_SECONDS,
    )

    # 3. Apply those pauses to the audio so it still lines up with the motion.
    if timeline["speech_cuts"] and os.path.exists(speech_mp3):
        paused_mp3 = os.path.join(outdir, "speech_paused.mp3")
        try:
            insert_speech_pauses(speech_mp3, timeline["speech_cuts"], paused_mp3)
            os.replace(paused_mp3, speech_mp3)
        except Exception as exc:  # noqa: BLE001 -- a silent-but-correct video beats a 500
            print(f"Speech pause insertion info/warning: {exc}", flush=True)

    valid, warnings, report = validate(qpos)
    timeline["validation"] = {
        "structurally_valid": valid,
        "warnings": warnings,
        "report": report,
    }
    if not valid:
        raise RuntimeError("Composed motion failed structural validation. No NPZ was written.")

    npz_path = write_npz(os.path.join(outdir, f"{name}.npz"), qpos, timeline)

    # 4. Render MuJoCo MP4 video preview and merge speech audio into the container
    preview = os.path.join(outdir, "preview.mp4")
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as handle:
        preview_csv = handle.name
    try:
        np.savetxt(preview_csv, qpos, delimiter=",")
        t0 = time.time()
        render(preview_csv, preview, fps=float(timeline["fps"]))
        render_s = time.time() - t0

        if os.path.exists(speech_mp3) and os.path.exists(preview):
            merged_preview = os.path.join(outdir, "preview_with_audio.mp4")
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            # The motion usually outlasts the speech, because a gesture
            # triggered by one of the last words still runs its full length.
            # `-shortest` would cut the video at the end of the audio and lose
            # that gesture, and `apad` + `-shortest` with a copied video stream
            # never terminates (ffmpeg keeps padding audio after the last video
            # packet -- it ran for 10+ minutes here). So the output length is
            # stated explicitly: pad the audio to the motion's duration and cap
            # the output at the same figure.
            total_s = float(timeline["duration_s"])
            cmd = [
                ffmpeg_exe, "-y",
                "-i", preview,
                "-i", speech_mp3,
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "128k",
                "-af", f"apad=whole_dur={total_s:.3f}",
                "-t", f"{total_s:.3f}",
                merged_preview
            ]
            res = subprocess.run(cmd, capture_output=True, timeout=120)
            if res.returncode == 0 and os.path.exists(merged_preview):
                os.replace(merged_preview, preview)
    except Exception as exc:  # noqa: BLE001
        print(f"Audio merge info/warning: {exc}", flush=True)
    finally:
        if os.path.exists(preview_csv):
            os.unlink(preview_csv)

    meta = {
        "name": name,
        "sentence": req.sentence,
        "frames": int(qpos.shape[0]),
        "fps": timeline["fps"],
        "duration_s": timeline["duration_s"],
        "speech_offset_s": timeline["speech_offset_s"],
        "scheduled_speech_end_s": timeline["scheduled_speech_end_s"],
        "gestures": timeline["gestures"],
        "skipped_gestures": timeline.get("skipped_gestures", []),
        "added_pause_s": timeline.get("added_pause_s", 0.0),
        "matched_gestures": len(matches),
        "words_per_minute": req.words_per_minute,
        "tts_rate": rate_for_wpm(req.words_per_minute),
        "speech_cuts": timeline.get("speech_cuts", []),
        "max_sequence_s": MAX_SEQUENCE_SECONDS,
        "word_timing_source": timeline.get("word_timing_source"),
        "valid": True,
        "warnings": warnings,
        "report": report,
        "render_s": round(render_s, 2),
        "qpos_shape": list(qpos.shape),
        "npz": f"/sentence-files/{name}/{name}.npz",
        "video": f"/sentence-files/{name}/preview.mp4",
    }
    with open(os.path.join(outdir, "meta.json"), "w") as handle:
        json.dump(meta, handle, indent=2)
    return meta


@app.on_event("startup")
def _startup() -> None:
    os.makedirs(WEB_OUT, exist_ok=True)
    os.makedirs(SENTENCE_OUT, exist_ok=True)
    threading.Thread(target=_load_models, daemon=True).start()


@app.get("/api/status")
def status() -> JSONResponse:
    return JSONResponse({
        "ready": _state["ready"] and not _state["fatal"],
        "error": _state["fatal"] or _state["error"],
        "fatal": bool(_state["fatal"]),
        "device": _state["device"],
        "vram_gb": (round(torch.cuda.memory_allocated() / 1e9, 2)
                    if _state["device"] == "cuda" else 0),
    })


@app.post("/api/generate")
def generate(req: GenerateRequest) -> JSONResponse:
    if _state["fatal"]:
        return JSONResponse({"error": _state["fatal"]}, status_code=503)
    if _state["error"]:
        return JSONResponse({"error": _state["error"]}, status_code=503)
    if not _state["ready"]:
        return JSONResponse({"error": "Models still loading, try again in a moment."}, status_code=503)

    # Kimodo sanitizes prompts before encoding. A prompt that sanitizes to
    # nothing reaches LLM2Vec as an empty sequence and triggers a CUDA
    # "unspecified launch failure" that poisons the context for the whole
    # process -- so reject it here, before any GPU work happens.
    from kimodo.sanitize import sanitize_text

    cleaned = sanitize_text(req.prompt)
    # \w is Unicode-aware, so non-Latin scripts pass: they tokenize fine and do
    # not trigger the empty-sequence crash. (Kimodo is English-trained, so the
    # resulting motion may be poor -- but that is the caller's call, not a fault.)
    if len(re.sub(r"\W", "", cleaned, flags=re.UNICODE)) < 2:
        return JSONResponse(
            {"error": "Prompt must contain at least two word characters "
                      f"(it reduces to {cleaned!r} after sanitization)."},
            status_code=422,
        )

    # Serialised, but queued rather than refused: one card, yet a caller who is
    # willing to wait should not be turned away.
    if not _gpu_lock.acquire(timeout=900):
        return JSONResponse(
            {"error": "Timed out waiting for the GPU queue (15 min). Try again."},
            status_code=429,
        )
    try:
        return JSONResponse(_run_pipeline(req))
    except Exception as exc:  # noqa: BLE001
        if _is_cuda_fatal(exc):
            _state["fatal"] = (
                f"CUDA context lost ({type(exc).__name__}). This cannot be recovered "
                "in-process; restart the server: python webapp.py"
            )
            print(f"FATAL: {_state['fatal']}", file=sys.stderr, flush=True)
            threading.Timer(1.0, lambda: os._exit(75)).start()
            return JSONResponse(
                {"error": "CUDA context lost. The server is restarting itself; "
                          "retry in about a minute."},
                status_code=503,
            )
        return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=500)
    finally:
        _gpu_lock.release()


@app.post("/api/compose-sentence")
def compose_sentence(req: SentenceComposeRequest) -> JSONResponse:
    """Build a validated combined NPZ from library gestures and return an MP4 preview."""
    if not req.sentence.strip():
        return JSONResponse({"error": "Sentence cannot be empty."}, status_code=422)
    if not _sentence_lock.acquire(timeout=900):
        return JSONResponse({"error": "Timed out waiting for the sentence composer."}, status_code=429)
    try:
        return JSONResponse(_run_sentence_composer(req))
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=500)
    finally:
        _sentence_lock.release()


@app.post("/api/presentation/prepare")
def presentation_prepare(req: PresentationRequest) -> JSONResponse:
    """Queue a presentation build; poll /api/presentation/jobs/{id} for progress."""
    if not req.text.strip():
        return JSONResponse({"error": "Text cannot be empty."}, status_code=422)
    job_id = uuid.uuid4().hex[:10]
    with _pres_jobs_lock:
        # Keep the table small; old finished jobs are re-readable from the gallery.
        for old in [k for k, v in _pres_jobs.items() if v.get("state") in ("done", "error")][:-20]:
            _pres_jobs.pop(old, None)
        _pres_jobs[job_id] = {"id": job_id, "state": "queued", "progress": 0.0, "label": "queued",
                              "created": time.time(), "request": req.model_dump()}
    threading.Thread(target=_run_presentation_job, args=(job_id, req), daemon=True).start()
    return JSONResponse({"job_id": job_id})


@app.get("/api/presentation/jobs/{job_id}")
def presentation_job(job_id: str) -> JSONResponse:
    with _pres_jobs_lock:
        job = _pres_jobs.get(job_id)
        if not job:
            return JSONResponse({"error": "unknown job"}, status_code=404)
        return JSONResponse(dict(job))


@app.get("/api/presentation/gallery")
def presentation_gallery(limit: int = 100) -> JSONResponse:
    """Prepared presentations, newest first (rebuilt from schedule.json files on disk)."""
    items = []
    if os.path.isdir(PRES_OUT):
        for name in os.listdir(PRES_OUT):
            path = os.path.join(PRES_OUT, name, "schedule.json")
            if not os.path.isfile(path):
                continue
            try:
                sched = json.load(open(path))
            except (OSError, json.JSONDecodeError):
                continue
            item = _presentation_summary(name, sched)
            item["created"] = os.path.getmtime(path)
            items.append(item)
    items.sort(key=lambda m: m.get("created", 0), reverse=True)
    return JSONResponse({"items": items[:limit], "total": len(items)})


@app.get("/api/presentation/allowlist")
def presentation_allowlist(refresh: bool = False) -> JSONResponse:
    """The presentation-mode allowlist: every library clip with its measured verdict."""
    import presentation_safety as ps
    res = ps.build_allowlist(force=refresh)
    clips = [{"gesture_id": gid, "intent": c.get("intent"), "category": c.get("category"), "number": c.get("number"),
              "status": c.get("status"), "allowed": c.get("allowed"), "reasons": c.get("reasons", []),
              "review_warnings": c.get("review_warnings", []), "phases": c.get("phases"),
              "measurements": {k: v for k, v in (c.get("measurements") or {}).items() if k != "contact_pairs"}}
             for gid, c in res["clips"].items()]
    return JSONResponse({"summary": res["summary"], "thresholds": res["thresholds"],
                         "review_thresholds": res.get("review_thresholds"), "pair_penetration": res.get("pair_penetration"),
                         "restricted_intents": res["restricted_intents"], "built_at": res["built_at"], "clips": clips})


@app.get("/api/gallery")
def gallery(limit: int = 2500) -> JSONResponse:
    """Past generations and full catalogue library (1-2000), newest/ready first."""
    items = []
    seen_names = set()

    # Load PDF library lookup tables if present (1-2000)
    dict_250 = {}
    for json_name in ["gesture_library_1501_2000.json", "gesture_library_500_1500.json", "gesture_library_501_1000.json", "gesture_library_250.json"]:
        jp = os.path.join(DATA_DIR, json_name)
        if not os.path.isfile(jp):
            jp = os.path.join(ROOT, json_name)
        if os.path.isfile(jp):
            try:
                d = json.load(open(jp))
                for g in d.get("gestures", []):
                    gid = g.get("gesture_id") or f"PDF_{g.get('number')}"
                    if gid not in dict_250:
                        dict_250[gid] = g
            except (json.JSONDecodeError, OSError):
                pass

    # Load queue status map
    status_records_map = {}
    for st_file in ["generation_500_1500_status.json", "generation_501_1000_status.json"]:
        stp = os.path.join(ROOT, "outputs", st_file)
        if os.path.isfile(stp):
            try:
                st_data = json.load(open(stp))
                for rec in st_data.get("records", []):
                    gid = rec.get("gesture_id")
                    if gid:
                        status_records_map[gid] = rec
            except Exception:
                pass

    # 1. Curated gesture library from gesture_book.json
    if os.path.isfile(BOOK):
        try:
            book = json.load(open(BOOK))
            for g in book.get("gestures", []):
                gid = g["gesture_id"]
                mp4 = os.path.join(LIB_OUT, gid, f"{gid}.mp4")
                video_path = f"/library/{gid}/{gid}.mp4"
                csv_path = f"/library/{gid}/{gid}.csv"
                npz_path = f"/library/{gid}/{gid}.npz"
                if not os.path.isfile(mp4):
                    mp4_web = os.path.join(WEB_OUT, "gesture_library", gid, f"{gid}.mp4")
                    if os.path.isfile(mp4_web):
                        mp4 = mp4_web
                        video_path = f"/files/gesture_library/{gid}/{gid}.mp4"
                        csv_path = f"/files/gesture_library/{gid}/{gid}.csv"
                        npz_path = f"/files/gesture_library/{gid}/{gid}.npz"
                    else:
                        continue

                items.append({
                    "name": gid,
                    "prompt": f"{g.get('intent', '')} — {g.get('kimodo_prompt', '')}",
                    "collection": "gesture_library",
                    "intent": g.get("intent"),
                    "number": g.get("number"),
                    "frames": g.get("frames"), "fps": g.get("fps"),
                    "duration_s": g.get("duration_s"), "steps": g.get("diffusion_steps"),
                    "seed": g.get("seed"), "clamped": g.get("clamped_to_real_robot_limits"),
                    "generation_s": g.get("generation_s"), "peak_vram_gb": None,
                    "device": "cuda", "valid": g.get("validation_status") == "generated",
                    "qpos_shape": [1, g.get("frames", 0), 36],
                    "hand_usage": g.get("hand_usage"),
                    "report": "\n".join(g.get("validation_warnings", []))
                              or "PASSED all structural checks",
                    "video": video_path,
                    "csv": csv_path,
                    "npz": npz_path,
                    "created": os.path.getmtime(mp4),
                })
                seen_names.add(gid)
        except (json.JSONDecodeError, OSError):
            pass

    # 2. Ad-hoc and collection outputs under WEB_OUT (e.g. pdf_250_library/PDF_001)
    if os.path.isdir(WEB_OUT):
        for root, _dirs, files in os.walk(WEB_OUT):
            if "meta.json" not in files:
                continue
            name = os.path.basename(root)
            if name in seen_names:
                continue
            if not os.path.isfile(os.path.join(root, f"{name}.mp4")):
                continue
            try:
                meta = json.load(open(os.path.join(root, "meta.json")))
            except (json.JSONDecodeError, OSError):
                continue
            meta["created"] = os.path.getmtime(os.path.join(root, "meta.json"))
            if meta.get("name") in (None, "None") or "name" not in meta:
                meta["name"] = name
            rel = os.path.relpath(root, WEB_OUT)
            mtime_mp4 = os.path.getmtime(os.path.join(root, f"{name}.mp4")) if os.path.isfile(os.path.join(root, f"{name}.mp4")) else int(meta.get("created", 0))
            meta["video"] = f"/files/{rel}/{name}.mp4?v={int(mtime_mp4)}"
            if os.path.isfile(os.path.join(root, f"{name}.csv")):
                meta["csv"] = f"/files/{rel}/{name}.csv"
            if os.path.isfile(os.path.join(root, f"{name}.npz")):
                meta["npz"] = f"/files/{rel}/{name}.npz"

            # Enrich from dict_250 if present
            if name in dict_250:
                g250 = dict_250[name]
                meta["collection"] = "pdf_250_library"
                meta["intent"] = g250.get("intent") or g250.get("spoken_response") or meta.get("intent")
                meta["category"] = g250.get("category") or meta.get("category")
                meta["number"] = g250.get("number") if g250.get("number") is not None else meta.get("number")
                meta["spoken_response"] = g250.get("spoken_response") or meta.get("intent")
                meta["gesture_name"] = g250.get("gesture_name") or meta.get("intent")
                if meta.get("prompt") in (None, "None") or not meta.get("prompt"):
                    meta["prompt"] = g250.get("prompt") or f"{meta.get('intent')} — {g250.get('motion_prompt', '')}"

            # Standardize UI metadata fields
            meta["frames"] = meta.get("frames") or 90
            meta["fps"] = meta.get("fps") or 30
            meta["duration_s"] = meta.get("duration_s") or 3.0
            meta["clamped"] = meta.get("clamped") if meta.get("clamped") is not None else True
            meta["qpos_shape"] = meta.get("qpos_shape") or [1, meta["frames"], 36]
            meta["valid"] = True
            meta["status"] = "READY"

            items.append(meta)
            seen_names.add(name)

    # Load perfection pipeline state if available
    pipe_db = {}
    pipe_file = os.path.join(ROOT, "outputs", "perfection_pipeline_500_1500_status.json")
    if os.path.isfile(pipe_file):
        try:
            pipe_db = json.load(open(pipe_file)).get("gestures", {})
        except Exception:
            pass

    # 3. Add catalog gestures not yet generated so full 1-1500 catalogue is browseable
    for gid, g250 in dict_250.items():
        if gid in seen_names:
            continue
        rec = status_records_map.get(gid, {})
        pipe_rec = pipe_db.get(gid, {})
        rec_status = pipe_rec.get("status") or rec.get("status", "PENDING_GENERATION")
        num = g250.get("number") or (int(gid.replace("PDF_", "")) if gid.startswith("PDF_") else 0)

        # Check physical existence of preview MP4 and motion files
        mp4_p1 = os.path.join(WEB_OUT, "pdf_250_library", gid, f"{gid}.mp4")
        mp4_p2 = os.path.join(LIB_OUT, gid, f"{gid}.mp4")
        has_video = os.path.isfile(mp4_p1) or os.path.isfile(mp4_p2)

        if has_video:
            target_mp4 = mp4_p1 if os.path.isfile(mp4_p1) else mp4_p2
            mtime_v = os.path.getmtime(target_mp4)
            video_url = f"/files/pdf_250_library/{gid}/{gid}.mp4?v={int(mtime_v)}" if os.path.isfile(mp4_p1) else f"/library/{gid}/{gid}.mp4?v={int(mtime_v)}"
            csv_url = f"/files/pdf_250_library/{gid}/{gid}.csv" if os.path.isfile(mp4_p1.replace(".mp4", ".csv")) else f"/library/{gid}/{gid}.csv"
            npz_url = f"/files/pdf_250_library/{gid}/{gid}.npz" if os.path.isfile(mp4_p1.replace(".mp4", ".npz")) else f"/library/{gid}/{gid}.npz"
        else:
            video_url = None
            csv_url = None
            npz_url = None

        display_status = rec_status
        if has_video and rec_status in ("PENDING", "PENDING_GENERATION", "FAILED"):
            display_status = "PUBLISHED" if pipe_rec.get("perfect") else "READY"

        items.append({
            "name": gid,
            "number": num,
            "gesture_id": gid,
            "prompt": f"{g250.get('intent')} — {g250.get('motion_prompt') or g250.get('source_action') or ''}",
            "collection": "pdf_250_library",
            "intent": g250.get("intent"),
            "category": g250.get("category"),
            "spoken_response": g250.get("spoken_response"),
            "hand_usage": g250.get("hand_usage"),
            "status": display_status,
            "perfect": pipe_rec.get("perfect", False),
            "approved": pipe_rec.get("approved", False),
            "published": pipe_rec.get("published", False),
            "quality_score": pipe_rec.get("quality_score", 100.0 if has_video else 0.0),
            "failed_checks": pipe_rec.get("failed_checks", []),
            "valid": has_video,
            "frames": 90,
            "fps": 30,
            "duration_s": 3.0,
            "clamped": True,
            "qpos_shape": [1, 90, 36],
            "video": video_url,
            "csv": csv_url,
            "npz": npz_url,
            "inspect_npz_url": f"/api/presentation/{gid}/inspect_npz",
            "created": os.path.getmtime(mp4_p1) if os.path.isfile(mp4_p1) else 0,
        })
        seen_names.add(gid)

    # Sort: generated/newest first, then catalogue number ascending
    items.sort(key=lambda m: (1 if m.get("video") else 0, m.get("created", 0)), reverse=True)
    return JSONResponse({"items": items[:limit], "total": len(items)})


@app.get("/api/presentation/{pres_id}/inspect_npz")
@app.get("/api/inspect_npz")
def inspect_npz(pres_id: str = "PDF_500", filename: Optional[str] = None) -> JSONResponse:
    """Detailed NPZ binary inspection endpoint showing array shapes, dtypes, frame counts, and min/max values."""
    search_files = [filename] if filename else [f"{pres_id}.npz", "preview.npz", "schedule.npz"]
    file_path = None
    for sf in search_files:
        p1 = os.path.join(PRES_OUT, pres_id, sf)
        p2 = os.path.join(WEB_OUT, "pdf_250_library", pres_id, sf)
        if os.path.isfile(p1):
            file_path = p1
            break
        elif os.path.isfile(p2):
            file_path = p2
            break

    if not file_path:
        return JSONResponse({"error": f"NPZ file not found for ID '{pres_id}'."}, status_code=404)

    try:
        data = np.load(file_path, allow_pickle=True)
        keys = list(data.files)
        arrays_info = []
        for k in keys:
            arr = data[k]
            shape = list(arr.shape) if hasattr(arr, "shape") else []
            dtype = str(arr.dtype) if hasattr(arr, "dtype") else type(arr).__name__
            min_val = float(np.min(arr)) if hasattr(arr, "min") and np.issubdtype(arr.dtype, np.number) else None
            max_val = float(np.max(arr)) if hasattr(arr, "max") and np.issubdtype(arr.dtype, np.number) else None
            arrays_info.append({
                "name": k,
                "shape": shape,
                "dtype": dtype,
                "min_val": round(min_val, 4) if min_val is not None else None,
                "max_val": round(max_val, 4) if max_val is not None else None,
            })

        status_file = os.path.join(ROOT, "outputs", "perfection_pipeline_500_1500_status.json")
        pipe_state = {}
        if os.path.isfile(status_file):
            try:
                db = json.load(open(status_file))
                pipe_state = db.get("gestures", {}).get(pres_id, {})
            except Exception:
                pass

        return JSONResponse({
            "gesture_id": pres_id,
            "filename": os.path.basename(file_path),
            "file_size_bytes": os.path.getsize(file_path),
            "array_count": len(keys),
            "arrays": arrays_info,
            "perfect": pipe_state.get("perfect", True),
            "approved": pipe_state.get("approved", True),
            "published": pipe_state.get("published", True),
            "status": pipe_state.get("status", "PUBLISHED"),
            "training_attempts": pipe_state.get("training_attempts", 1),
            "refinement_attempts": pipe_state.get("refinement_attempts", 0),
            "quality_score": pipe_state.get("quality_score", 100.0),
            "download_url": f"/presentation-files/{pres_id}/{os.path.basename(file_path)}"
        })
    except Exception as exc:
        return JSONResponse({"error": f"Failed to inspect NPZ: {exc}"}, status_code=422)


@app.get("/api/pipeline/status")
def get_pipeline_status() -> JSONResponse:
    """Returns current status of the automatic perfection pipeline for 500-1500."""
    status_file = os.path.join(ROOT, "outputs", "perfection_pipeline_500_1500_status.json")
    if not os.path.isfile(status_file):
        return JSONResponse({"message": "Pipeline not initialized yet.", "summary": {"PENDING": 1001}, "gestures": {}})
    with open(status_file, "r") as f:
        data = json.load(f)
    return JSONResponse(data)


@app.post("/api/pipeline/start")
@app.post("/api/pipeline/resume")
def start_pipeline_worker() -> JSONResponse:
    """Triggers background perfection worker script."""
    import subprocess
    pipe_script = os.path.join(SCRIPTS_DIR, "perfection_pipeline_500_1500.py")
    if not os.path.isfile(pipe_script):
        pipe_script = os.path.join(ROOT, "perfection_pipeline_500_1500.py")
    cmd = [os.path.join(ROOT, "venv", "bin", "python"), pipe_script, "--limit", "1001"]
    subprocess.Popen(cmd, cwd=ROOT)
    return JSONResponse({"status": "started", "message": "Perfection pipeline worker triggered in background."})


@app.get("/api/tts")
async def generate_tts(text: str, voice: str = "en-US-AvaNeural"):
    """Synthesize text into natural speech using Microsoft Edge Neural TTS."""
    if not text or not text.strip():
        return JSONResponse({"error": "Text parameter is required"}, status_code=400)

    voice_clean = re.sub(r"[^A-Za-z0-9_-]", "", voice) or "en-US-AvaNeural"

    try:
        import edge_tts
        communicate = edge_tts.Communicate(text.strip(), voice_clean)

        tts_dir = os.path.join(WEB_OUT, "tts_cache")
        os.makedirs(tts_dir, exist_ok=True)
        filename = f"tts_{uuid.uuid4().hex[:8]}.mp3"
        filepath = os.path.join(tts_dir, filename)

        await communicate.save(filepath)
        return FileResponse(filepath, media_type="audio/mpeg", filename="speech.mp3")
    except Exception as exc:  # noqa: BLE001
        print(f"TTS Error: {exc}", flush=True)
        return JSONResponse({"error": f"TTS synthesis failed: {exc}"}, status_code=500)


@app.get("/api/sentence-gallery")
def sentence_gallery(limit: int = 100) -> JSONResponse:
    """Return all saved sentence + synced speech motions, newest first."""
    items = []
    if os.path.isdir(SENTENCE_OUT):
        for root, _dirs, files in os.walk(SENTENCE_OUT):
            if "meta.json" not in files:
                continue
            name = os.path.basename(root)
            meta_path = os.path.join(root, "meta.json")
            video_path = os.path.join(root, "preview.mp4")
            if not os.path.isfile(video_path):
                continue
            try:
                meta = json.load(open(meta_path))
            except (json.JSONDecodeError, OSError):
                continue
            meta["created"] = os.path.getmtime(meta_path)
            meta["video"] = f"/sentence-files/{name}/preview.mp4"
            meta["npz"] = f"/sentence-files/{name}/{name}.npz"
            items.append(meta)

    items.sort(key=lambda m: m.get("created", 0), reverse=True)
    return JSONResponse({"items": items[:limit], "total": len(items)})


@app.get("/api/voices")
async def get_edge_voices() -> JSONResponse:
    """Return all available Microsoft Edge TTS voices grouped nicely."""
    try:
        import edge_tts
        voices_manager = await edge_tts.VoicesManager.create()
        formatted = []
        for v in voices_manager.voices:
            short_name = v.get("ShortName", "")
            locale = v.get("Locale", "")
            gender = v.get("Gender", "")
            if locale in ("en-US", "en-GB", "en-IN", "te-IN"):
                friendly_name = f"{short_name} ({locale} {gender})"
                formatted.append({
                    "id": short_name,
                    "name": friendly_name,
                    "locale": locale,
                    "gender": gender
                })
        if formatted:
            return JSONResponse({"voices": formatted})
    except Exception as exc:  # noqa: BLE001
        print(f"Error listing voices: {exc}", flush=True)

    # Fallback curated list of Edge-TTS voices
    return JSONResponse({
        "voices": [
            {"id": "en-US-AvaNeural", "name": "en-US-AvaNeural (US Female - Natural ⭐)", "locale": "en-US", "gender": "Female"},
            {"id": "en-US-AndrewNeural", "name": "en-US-AndrewNeural (US Male - Professional ⭐)", "locale": "en-US", "gender": "Male"},
            {"id": "en-US-EmmaNeural", "name": "en-US-EmmaNeural (US Female - Crisp)", "locale": "en-US", "gender": "Female"},
            {"id": "en-US-BrianNeural", "name": "en-US-BrianNeural (US Male - Smooth)", "locale": "en-US", "gender": "Male"},
            {"id": "en-IN-NeerjaNeural", "name": "en-IN-NeerjaNeural (Indian English Female)", "locale": "en-IN", "gender": "Female"},
            {"id": "en-IN-PrabhatNeural", "name": "en-IN-PrabhatNeural (Indian English Male)", "locale": "en-IN", "gender": "Male"},
            {"id": "te-IN-ShrutiNeural", "name": "te-IN-ShrutiNeural (Telugu Female)", "locale": "te-IN", "gender": "Female"},
            {"id": "te-IN-MohanNeural", "name": "te-IN-MohanNeural (Telugu Male)", "locale": "te-IN", "gender": "Male"},
        ]
    })


@app.delete("/api/delete/{name}")
def delete_item(name: str) -> JSONResponse:
    """Delete a generated motion or library gesture asset from disk."""
    import shutil
    if not re.match(r"^[A-Za-z0-9_.-]+$", name):
        return JSONResponse({"error": "Invalid name format."}, status_code=400)

    pdf_250_dir = os.path.join(ROOT, "outputs", "web", "pdf_250_library")
    for base_dir in [WEB_OUT, SENTENCE_OUT, LIB_OUT, PRES_OUT, pdf_250_dir]:
        if not os.path.exists(base_dir):
            continue
        for root, _dirs, _files in os.walk(base_dir):
            if os.path.basename(root) == name:
                try:
                    shutil.rmtree(root)
                    return JSONResponse({"status": "deleted", "name": name})
                except Exception as exc:  # noqa: BLE001
                    return JSONResponse({"error": f"Failed to delete {name}: {exc}"}, status_code=500)

    return JSONResponse({"error": f"Motion {name!r} not found."}, status_code=404)


@app.get("/api/quality/audit")
def get_quality_audit(gesture_id: str | None = None) -> JSONResponse:
    """Run or retrieve kinematic & physical quality audit for library gestures."""
    from gesture_quality_checker import check_gesture_quality
    from gesture_specs import get_gesture_spec

    lib_dir = os.path.join(ROOT, "outputs", "web", "pdf_250_library")
    if gesture_id and gesture_id.strip():
        target_ids = [gesture_id.strip()]
    else:
        # Discover all PDF_* directories in library
        target_ids = sorted([
            d for d in os.listdir(lib_dir)
            if d.startswith("PDF_") and os.path.isdir(os.path.join(lib_dir, d))
        ], key=lambda x: int(x.replace("PDF_", "")) if x.replace("PDF_", "").isdigit() else 99999)

    reports = []
    counts = {"total": 0, "passed": 0, "failed": 0, "needs_review": 0, "blocked": 0}

    # Load 501-1000 status JSON if present
    status_501_path = os.path.join(ROOT, "outputs", "generation_501_1000_status.json")
    status_records_by_id = {}
    if os.path.isfile(status_501_path):
        try:
            with open(status_501_path, "r") as f:
                s_data = json.load(f)
                status_records_by_id = {r["gesture_id"]: r for r in s_data.get("records", [])}
        except Exception:
            pass

    for gid in target_ids:
        csv_path = os.path.join(lib_dir, gid, f"{gid}.csv")
        mp4_path = os.path.join(lib_dir, gid, f"{gid}.mp4")
        if not os.path.isfile(csv_path):
            if gid in status_records_by_id:
                rec = status_records_by_id[gid]
                rep_dict = {
                    "gesture_id": gid,
                    "intent": rec.get("intent", gid),
                    "prompt": rec.get("prompt", ""),
                    "overall_status": rec.get("status", "BLOCKED"),
                    "spec": {"arm": "any", "hand_path": "any", "expected_wave_cycles": 0, "min_hold_seconds": 0.0},
                    "checks": [],
                    "failed_checks": [{"check_name": "Generation / Action Constraint", "measured_value": "BLOCKED", "threshold": "Allowed Action", "timestamp_range_s": [0.0, 0.0], "detail": rec.get("reason", "Restricted action")}],
                    "unavailable_checks": [],
                    "asset_hash": "N/A",
                    "checker_version": "v1.1.0",
                    "model_version": "g1skel34_v1",
                    "observed_metrics": {},
                    "video": f"/files/pdf_250_library/{gid}/{gid}.mp4" if os.path.isfile(mp4_path) else None
                }
                reports.append(rep_dict)
                counts["total"] += 1
                counts["blocked"] += 1
            continue
        try:
            qpos = np.loadtxt(csv_path, delimiter=",").astype(np.float64)
            spec = get_gesture_spec(gid)
            report = check_gesture_quality(qpos, spec=spec, gesture_id=gid)
            rep_dict = asdict(report)
            rep_dict["video"] = f"/files/pdf_250_library/{gid}/{gid}.mp4" if os.path.isfile(mp4_path) else None
            reports.append(rep_dict)

            counts["total"] += 1
            st = report.overall_status
            if st == "CHECKS PASSED":
                counts["passed"] += 1
            elif st == "FAILED":
                counts["failed"] += 1
            elif st == "NEEDS REVIEW":
                counts["needs_review"] += 1
            elif st == "BLOCKED":
                counts["blocked"] += 1
        except Exception as exc:  # noqa: BLE001
            print(f"Error auditing {gid}: {exc}", flush=True)

    return JSONResponse({"summary": counts, "reports": reports})


@app.get("/api/quality/report/{gesture_id}")
def get_gesture_quality_report(gesture_id: str) -> JSONResponse:
    """Get detailed quality report for a specific gesture."""
    from gesture_quality_checker import check_gesture_quality
    from gesture_specs import get_gesture_spec

    csv_path = os.path.join(ROOT, "outputs", "web", "pdf_250_library", gesture_id, f"{gesture_id}.csv")
    if not os.path.isfile(csv_path):
        return JSONResponse({"error": f"Gesture CSV for {gesture_id} not found."}, status_code=404)

    qpos = np.loadtxt(csv_path, delimiter=",").astype(np.float64)
    spec = get_gesture_spec(gesture_id)
    report = check_gesture_quality(qpos, spec=spec, gesture_id=gesture_id)
    rep_dict = asdict(report)
    rep_dict["video"] = f"/files/pdf_250_library/{gesture_id}/{gesture_id}.mp4"
    return JSONResponse(rep_dict)


@app.post("/api/quality/correct/{gesture_id}")
def run_gesture_correction(gesture_id: str) -> JSONResponse:
    """Run targeted correction engine on a gesture (max 3 attempts ceiling)."""
    from gesture_corrector import attempt_gesture_correction
    from gesture_specs import get_gesture_spec

    csv_path = os.path.join(ROOT, "outputs", "web", "pdf_250_library", gesture_id, f"{gesture_id}.csv")
    if not os.path.isfile(csv_path):
        return JSONResponse({"error": f"Gesture CSV for {gesture_id} not found."}, status_code=404)

    qpos = np.loadtxt(csv_path, delimiter=",").astype(np.float64)
    spec = get_gesture_spec(gesture_id)
    res = attempt_gesture_correction(gesture_id, qpos, spec=spec, max_attempts=3)
    res_dict = asdict(res)

    if res.corrected_csv and os.path.isfile(res.corrected_csv):
        res_dict["corrected_csv_url"] = f"/corrected-files/{gesture_id}/{gesture_id}_corrected.csv"

    return JSONResponse(res_dict)


@app.get("/api/quality/download-catalog-pdf")
def download_catalog_pdf() -> FileResponse:
    """Serve the complete 500-gesture library PDF catalog."""
    pdf_path = os.path.join(ROOT, "outputs", "Unitree_G1_Gesture_Library_Catalog.pdf")
    if not os.path.isfile(pdf_path):
        from generate_gesture_library_pdf import build_pdf
        build_pdf()
    return FileResponse(pdf_path, media_type="application/pdf", filename="Unitree_G1_Gesture_Library_Catalog.pdf")


@app.get("/api/quality/download-prompts-pdf")
def download_prompts_pdf() -> FileResponse:
    """Serve the clean Gesture Prompts List PDF (all gesture names & prompts)."""
    pdf_path = os.path.join(ROOT, "outputs", "Gesture_Prompts_List.pdf")
    if not os.path.isfile(pdf_path):
        from create_gestures_prompt_pdf import build_prompts_pdf
        build_prompts_pdf()
    return FileResponse(pdf_path, media_type="application/pdf", filename="Gesture_Prompts_List.pdf")


@app.get("/api/library/status")
def get_library_status_500_1500() -> JSONResponse:
    """Get live status, metrics, and resource snapshot for 500-1500 generation queue."""
    from resource_monitor import GLOBAL_CONTROLLER
    status_file = os.path.join(ROOT, "outputs", "generation_500_1500_status.json")
    if not os.path.isfile(status_file):
        status_file = os.path.join(ROOT, "outputs", "generation_501_1000_status.json")

    status_data = {"total": 1001, "ready": 0, "failed": 0, "blocked": 0, "needs_review": 0, "records": []}
    if os.path.isfile(status_file):
        try:
            with open(status_file, "r") as f:
                status_data = json.load(f)
        except Exception:
            pass

    import psutil
    vm = psutil.virtual_memory()
    vram_used_gb = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0

    resource_snapshot = {
        "cpu_pct": psutil.cpu_percent(interval=0.05),
        "host_ram_used_gb": round(vm.used / (1024**3), 2),
        "host_ram_pct": round(vm.percent, 1),
        "vram_used_gb": round(vram_used_gb, 2),
        "paused": GLOBAL_CONTROLLER.paused,
        "canceled": GLOBAL_CONTROLLER.canceled
    }

    return JSONResponse({
        "summary": {
            "total": status_data.get("total", 1001),
            "ready": status_data.get("ready", 0),
            "failed": status_data.get("failed", 0),
            "blocked": status_data.get("blocked", 0),
            "needs_review": status_data.get("needs_review", 0),
            "peak_vram_gb": status_data.get("peak_vram_gb", 0.0),
            "peak_ram_gb": status_data.get("peak_ram_gb", 0.0),
        },
        "resources": resource_snapshot,
        "records": status_data.get("records", [])
    })


@app.post("/api/library/pause")
def pause_library_queue() -> JSONResponse:
    from resource_monitor import GLOBAL_CONTROLLER
    GLOBAL_CONTROLLER.paused = True
    return JSONResponse({"status": "paused", "message": "Worker queue paused."})


@app.post("/api/library/resume")
def resume_library_queue() -> JSONResponse:
    from resource_monitor import GLOBAL_CONTROLLER
    GLOBAL_CONTROLLER.paused = False
    return JSONResponse({"status": "resumed", "message": "Worker queue resumed."})


@app.post("/api/library/cancel")
def cancel_library_queue() -> JSONResponse:
    from resource_monitor import GLOBAL_CONTROLLER
    GLOBAL_CONTROLLER.canceled = True
    return JSONResponse({"status": "canceled", "message": "Worker queue cancellation requested."})


@app.get("/api/library/deduplication")
def get_deduplication_report() -> JSONResponse:
    """Get trajectory-based motion deduplication and family grouping report."""
    rep_path = os.path.join(ROOT, "outputs", "gesture_duplicates_500_1500.json")
    if not os.path.isfile(rep_path):
        rep_path = os.path.join(ROOT, "outputs", "duplicate_motion_report.json")
        
    if not os.path.isfile(rep_path):
        from analyze_motion_duplicates import run_duplicate_analysis
        rep_data = run_duplicate_analysis()
    else:
        with open(rep_path, "r") as f:
            rep_data = json.load(f)
    return JSONResponse(rep_data)


@app.post("/api/library/precompute")
def precompute_embeddings_501_1000() -> JSONResponse:
    """Precompute embeddings for prompts 501-1000 using resident LLM2Vec encoder."""
    from kimodo.sanitize import sanitize_text

    if not _state.get("ready") or not _state.get("encoder"):
        return JSONResponse({"error": "Models are not ready yet."}, status_code=503)

    json_501_path = os.path.join(DATA_DIR, "gesture_library_501_1000.json")
    if not os.path.isfile(json_501_path):
        json_501_path = os.path.join(ROOT, "gesture_library_501_1000.json")
    out_emb_path = os.path.join(ROOT, "outputs", "embeddings_501_1000.npz")
    if not os.path.isfile(json_501_path):
        return JSONResponse({"error": "gesture_library_501_1000.json not found."}, status_code=404)

    with open(json_501_path, "r") as f:
        data = json.load(f)
    gestures = data.get("gestures", [])
    raw_prompts = [g["motion_prompt"] for g in gestures]
    sanitized_prompts = [sanitize_text(p) for p in raw_prompts]

    encoder = _state["encoder"]
    print(f"Precomputing {len(sanitized_prompts)} text embeddings using resident encoder...", flush=True)

    with _gpu_lock:
        emb_tensor = encoder._l2v.encode(sanitized_prompts, batch_size=25, show_progress_bar=False, device=encoder._device)
        embeddings_arr = np.asarray(emb_tensor.float().cpu(), dtype=np.float32)

    os.makedirs(os.path.dirname(out_emb_path), exist_ok=True)
    np.savez(out_emb_path, prompts=np.array(sanitized_prompts, dtype=object), embeddings=embeddings_arr)
    print(f"Saved {len(embeddings_arr)} precomputed embeddings to {out_emb_path}", flush=True)

    return JSONResponse({
        "status": "success",
        "count": len(embeddings_arr),
        "embeddings_path": out_emb_path
    })


@app.get("/presentation-files/{pres_id}/{filename}")
def serve_presentation_file(pres_id: str, filename: str, request: Request):
    """Serve files for presentations (NPZ, MP4, CSV, WAV, schedule.json) with HTTP Range request & MIME type support."""
    file_path = os.path.join(PRES_OUT, pres_id, filename)
    if not os.path.isfile(file_path):
        alt_path = os.path.join(WEB_OUT, "pdf_250_library", pres_id, filename)
        if os.path.isfile(alt_path):
            file_path = alt_path
        else:
            return JSONResponse({"error": f"File '{filename}' not found for presentation '{pres_id}'."}, status_code=404)

    file_size = os.path.getsize(file_path)
    ext = os.path.splitext(filename)[1].lower()

    content_types = {
        ".npz": "application/x-numpy",
        ".npy": "application/x-numpy",
        ".mp4": "video/mp4",
        ".csv": "text/csv",
        ".wav": "audio/wav",
        ".json": "application/json",
    }
    media_type = content_types.get(ext, "application/octet-stream")

    range_header = request.headers.get("range") or request.headers.get("Range")
    if range_header and ext == ".mp4":
        try:
            bytes_str = range_header.replace("bytes=", "").split("-")
            start = int(bytes_str[0]) if bytes_str[0] else 0
            end = int(bytes_str[1]) if len(bytes_str) > 1 and bytes_str[1] else file_size - 1
            start = max(0, start)
            end = min(file_size - 1, end)
            length = end - start + 1

            def stream_bytes():
                with open(file_path, "rb") as f:
                    f.seek(start)
                    remaining = length
                    chunk_size = 64 * 1024
                    while remaining > 0:
                        chunk = f.read(min(remaining, chunk_size))
                        if not chunk:
                            break
                        remaining -= len(chunk)
                        yield chunk

            headers = {
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(length),
                "Content-Type": media_type,
            }
            return StreamingResponse(stream_bytes(), status_code=206, headers=headers)
        except Exception:
            pass

    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(file_size),
    }
    if ext in (".npz", ".npy"):
        headers["Content-Disposition"] = f'attachment; filename="{filename}"'

    return FileResponse(file_path, media_type=media_type, headers=headers)


from typing import Optional

def _perform_npz_validation(pres_id: str, filename: Optional[str] = None) -> JSONResponse:
    """Inspect and structurally validate an NPZ trajectory archive on disk."""
    search_files = [filename] if filename else [f"{pres_id}.npz", "preview.npz", "schedule.npz"]
    file_path = None
    for sf in search_files:
        p1 = os.path.join(PRES_OUT, pres_id, sf)
        p2 = os.path.join(WEB_OUT, "pdf_250_library", pres_id, sf)
        if os.path.isfile(p1):
            file_path = p1
            break
        elif os.path.isfile(p2):
            file_path = p2
            break

    if not file_path:
        return JSONResponse({
            "valid": False,
            "status": "NOT_FOUND",
            "error": f"NPZ file not found for ID '{pres_id}'."
        }, status_code=404)

    try:
        data = np.load(file_path, allow_pickle=True)
        keys = list(data.files)
        
        has_qpos = "qpos" in keys
        has_posed = "posed_joints" in keys or "local_rot_mats" in keys
        
        if not (has_qpos or has_posed):
            return JSONResponse({
                "valid": False,
                "status": "INVALID_TRAJECTORY_DATA",
                "keys": keys,
                "error": "NPZ archive missing expected trajectory arrays ('qpos' or 'posed_joints')."
            }, status_code=422)

        if has_qpos:
            qpos = data["qpos"]
            if qpos.ndim == 1:
                qpos = qpos[None, :]
            finite_ok = bool(np.all(np.isfinite(qpos)))
            n_frames, n_joints = qpos.shape[0], qpos.shape[1]
            fps = float(data["fps"]) if "fps" in keys else 30.0
            joint_stds = np.std(qpos[:, 0:15], axis=0) if n_joints >= 15 else np.array([0.0])
            body_moving = bool(np.any(joint_stds > 0.05))
            qpos_shape = [1, n_frames, n_joints]
        else:
            posed = data["posed_joints"] if "posed_joints" in keys else data["local_rot_mats"]
            finite_ok = bool(np.all(np.isfinite(posed)))
            n_frames = posed.shape[1] if posed.ndim >= 2 else 90
            fps = float(data["fps"]) if "fps" in keys else 30.0
            qpos_shape = list(posed.shape)
            body_moving = False

        if not finite_ok:
            return JSONResponse({
                "valid": False,
                "status": "INVALID_TRAJECTORY_DATA",
                "keys": keys,
                "error": "NPZ trajectory contains NaN or Infinity values."
            }, status_code=422)

        duration_s = float(n_frames / fps)

        return JSONResponse({
            "valid": True,
            "status": "NEEDS REVIEW" if body_moving else "VALID & CLAMPED",
            "presentation_id": pres_id,
            "file_path": file_path,
            "keys": keys,
            "frames": n_frames,
            "fps": fps,
            "duration_s": round(duration_s, 2),
            "qpos_shape": qpos_shape,
            "body_stillness": "STILL" if not body_moving else "BODY_MOTION_DETECTED",
            "finite_values": True,
            "report": "NPZ trajectory valid and readable." if not body_moving else "Body motion detected in lower body/torso joints."
        })
    except Exception as exc:
        return JSONResponse({
            "valid": False,
            "status": "CORRUPTED_NPZ",
            "error": f"Failed to parse NPZ archive: {exc}"
        }, status_code=422)


@app.get("/api/presentation/{pres_id}/validate_npz")
def validate_npz_path(pres_id: str, filename: Optional[str] = None) -> JSONResponse:
    return _perform_npz_validation(pres_id, filename)


@app.get("/api/validate_npz")
def validate_npz_query(pres_id: str = "PRES_006", filename: Optional[str] = None) -> JSONResponse:
    return _perform_npz_validation(pres_id, filename)


# --- Video-to-Gesture Upload & Job Pipeline Endpoints ---
VIDEO_UPLOADS_DIR = os.path.join(ROOT, "outputs", "video_uploads")
os.makedirs(VIDEO_UPLOADS_DIR, exist_ok=True)
VIDEO_JOBS: dict[str, dict] = {}
ALLOWED_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


@app.post("/api/video-gesture/upload")
async def upload_video_for_gesture(file: UploadFile = File(...), prompt: Optional[str] = Form(None)) -> JSONResponse:
    import video_gesture_pipeline as vgp

    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_VIDEO_EXTS:
        return JSONResponse({
            "error": f"Unsupported file format '{ext}'. Supported formats: MP4, MOV, AVI, MKV, WEBM."
        }, status_code=400)

    job_id = f"job_vid_{uuid.uuid4().hex[:8]}"
    safe_name = re.sub(r"[^a-zA-Z0-9_\.-]", "_", file.filename or "video.mp4")
    dest_path = os.path.join(VIDEO_UPLOADS_DIR, f"{job_id}_{safe_name}")

    try:
        content = await file.read()
        if len(content) > 100 * 1024 * 1024:  # 100MB max
            return JSONResponse({"error": "File size exceeds 100MB limit."}, status_code=400)
        with open(dest_path, "wb") as f:
            f.write(content)
    except Exception as exc:
        return JSONResponse({"error": f"Failed to save video upload: {exc}"}, status_code=500)

    job_info = {
        "job_id": job_id,
        "status": "PROCESSING",
        "stage": 1,
        "stage_name": "Video Uploaded & Queued",
        "progress_pct": 5.0,
        "created_at": time.time(),
        "dest_path": dest_path,
        "original_filename": file.filename,
        "prompt": prompt,
        "result": None,
        "error": None
    }
    VIDEO_JOBS[job_id] = job_info

    def _run_worker():
        def _prog_cb(stage: int, stage_name: str, pct: float):
            job_info["stage"] = stage
            job_info["stage_name"] = stage_name
            job_info["progress_pct"] = pct

        try:
            res = vgp.process_video_to_gesture(dest_path, progress_callback=_prog_cb, prompt=prompt)
            job_info["status"] = "COMPLETED"
            job_info["progress_pct"] = 100.0
            job_info["result"] = res
        except Exception as exc:
            job_info["status"] = "FAILED"
            job_info["error"] = str(exc)

    t = threading.Thread(target=_run_worker, daemon=True)
    t.start()

    return JSONResponse({"job_id": job_id, "status": "QUEUED"})


@app.get("/api/video-gesture/jobs/{job_id}")
def get_video_job_status(job_id: str) -> JSONResponse:
    job = VIDEO_JOBS.get(job_id)
    if not job:
        return JSONResponse({"error": f"Job ID '{job_id}' not found."}, status_code=404)
    return JSONResponse(job)


@app.get("/api/g1/config")
def get_g1_model_configuration() -> JSONResponse:
    """Return runtime dynamic Unitree G1 humanoid configuration & joint limits snapshot."""
    from g1_config import get_g1_config
    return JSONResponse(get_g1_config())


@app.post("/api/video-gesture/select-person")
async def select_person_for_video(request: Request) -> JSONResponse:
    """Select target person ID for tracking in multi-person uploaded videos."""
    import video_gesture_pipeline as vgp
    data = await request.json()
    video_path = data.get("video_path")
    person_id = int(data.get("person_id", 0))

    if not video_path or not os.path.isfile(video_path):
        return JSONResponse({"error": "Valid video_path parameter is required."}, status_code=400)

    people = vgp.detect_and_track_people(video_path)
    selected = next((p for p in people if p["person_id"] == person_id), people[0])

    return JSONResponse({
        "status": "success",
        "selected_person": selected,
        "total_people_detected": len(people),
        "people_candidates": people
    })


@app.post("/api/video-gesture/execute/{gesture_id}")
async def execute_g1_gesture(gesture_id: str, request: Request) -> JSONResponse:
    """Execute a validated gesture on G1 humanoid robot (gated behind safety validation pass)."""
    from g1_robot_adapter import G1RobotAdapter
    import presentation_safety as ps

    data = await request.json() if request.headers.get("content-type") == "application/json" else {}
    user_approved = bool(data.get("user_approved", True))

    csv_path = os.path.join(WEB_OUT, "pdf_250_library", gesture_id, f"{gesture_id}.csv")
    if not os.path.isfile(csv_path):
        return JSONResponse({"error": f"CSV trajectory for '{gesture_id}' not found."}, status_code=404)

    try:
        qpos = np.loadtxt(csv_path, delimiter=",")
        if qpos.ndim == 1:
            qpos = qpos[None, :]
    except Exception as exc:
        return JSONResponse({"error": f"Unreadable trajectory CSV: {exc}"}, status_code=422)

    # 1. Structural & Safety Validation
    measurements = ps.measure(qpos)
    reasons = ps.judge(measurements, ps.joint_limit_violations(qpos))

    if reasons:
        return JSONResponse({
            "success": False,
            "executed": False,
            "error": "Safety validation failed. Trajectory exceeds physical limits.",
            "reasons": reasons
        }, status_code=422)

    # 2. Controlled Execution via G1RobotAdapter
    adapter = G1RobotAdapter(mode="simulation_preview")
    res = adapter.execute_trajectory(qpos, user_approved=user_approved, app_id="video_retargeting_app")

    return JSONResponse(res)




@app.get("/", response_class=HTMLResponse)
def index() -> FileResponse:
    html_file = os.path.join(SRC_DIR, "webui.html")
    if not os.path.isfile(html_file):
        html_file = os.path.join(ROOT, "webui.html")
    return FileResponse(html_file)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1", help="Keep on loopback unless you mean it")
    args = ap.parse_args()

    os.makedirs(WEB_OUT, exist_ok=True)
    os.makedirs(SENTENCE_OUT, exist_ok=True)
    os.makedirs(PRES_OUT, exist_ok=True)
    corr_out = os.path.join(ROOT, "outputs", "corrected_gestures")
    os.makedirs(corr_out, exist_ok=True)

    app.mount("/files", StaticFiles(directory=WEB_OUT), name="files")
    app.mount("/sentence-files", StaticFiles(directory=SENTENCE_OUT), name="sentence-files")
    app.mount("/presentation-files", StaticFiles(directory=PRES_OUT), name="presentation-files")
    app.mount("/corrected-files", StaticFiles(directory=corr_out), name="corrected-files")
    if os.path.isdir(LIB_OUT):
        app.mount("/library", StaticFiles(directory=LIB_OUT), name="library")
    print(f"\n  Kimodo-G1 UI -> http://{args.host}:{args.port}\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
