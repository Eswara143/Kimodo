#!/usr/bin/env python3
"""Generate, validate and catalogue the whole G1 gesture library in one pass.

Reads gesture_library.json (Phase 2 definitions), then for every gesture:
    encode -> diffuse -> clamp to real-robot limits -> qpos CSV -> validate -> MP4

Writes gesture_book.json in the Phase 1 document's schema, with the fields that
can only be known after generation filled in from measurement rather than
assertion: hand_usage comes from actual per-joint travel, validation_status from
the validator's exit path.

The text encoder is loaded once for all prompts, then released before the
diffusion model loads -- same reason precompute/generate are split elsewhere.

    python build_gesture_library.py                 # everything
    python build_gesture_library.py --priority High # High-priority subset
    python build_gesture_library.py --only G001 G007
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(ROOT, "gesture_library.json")
BOOK = os.path.join(ROOT, "gesture_book.json")
OUT_ROOT = os.path.join(ROOT, "outputs", "gesture_library")
EMB = os.path.join(ROOT, "outputs", "gesture_library_embeddings.npz")
MERGED = os.path.expanduser("~/.cache/kimodo/llm2vec_merged")


def check_vram_free(need_gib: float = 5.0) -> None:
    """Fail early and clearly if something else already holds the GPU.

    The webapp keeps ~4.3 GB resident, which leaves too little for the encoder;
    without this the batch dies mid-way through loading with a raw CUDA OOM.
    """
    if not torch.cuda.is_available():
        return
    free, _ = torch.cuda.mem_get_info()
    if free / 1024**3 < need_gib:
        raise SystemExit(
            f"Only {free / 1024**3:.2f} GiB VRAM free; the encoder needs ~{need_gib:.0f} GiB.\n"
            "Something else is holding the GPU -- most likely the web backend.\n"
            "Stop it, run this, then start it again:\n"
            "    pkill -f start_backend.sh; pkill -f webapp.py\n"
            "    python build_gesture_library.py\n"
            "    ./start_backend.sh"
        )


def encode_all(prompts: list[str]) -> None:
    """One subprocess so the 15 GB encoder is fully released before Kimodo loads."""
    print(f"=== encoding {len(prompts)} prompts ===", flush=True)
    cmd = [sys.executable, os.path.join(ROOT, "precompute_text_embeddings.py"),
           *prompts, "--out", EMB, "--gpu-gib", "5", "--cpu-gib", "7"]
    proc = subprocess.run(cmd, cwd=ROOT)
    if proc.returncode != 0:
        raise SystemExit(f"encoding failed (exit {proc.returncode})")


def arm_usage(csv_path: str, model, names: dict[int, str]) -> tuple[str, float, float]:
    """Classify hand_usage from measured joint travel, not from the prompt text."""
    q = np.loadtxt(csv_path, delimiter=",")
    right = left = 0.0
    for col, name in names.items():
        if col >= q.shape[1]:
            continue
        if not any(k in name for k in ("shoulder", "elbow", "wrist")):
            continue
        travel = float(q[:, col].max() - q[:, col].min())
        if name.startswith("right_"):
            right += travel
        elif name.startswith("left_"):
            left += travel
    ratio = right / max(left, 1e-6)
    if ratio > 2.5:
        usage = "right"
    elif ratio < 0.4:
        usage = "left"
    elif right + left < 1.0:
        usage = "whole upper body"
    else:
        usage = "both"
    return usage, right, left


def build_via_api(lib: list[dict], args) -> int:
    """Drive a running backend. One model copy, no VRAM contention, web stays up."""
    import urllib.error
    import urllib.request

    base = args.via_api.rstrip("/")
    try:
        with urllib.request.urlopen(f"{base}/api/status", timeout=10) as r:
            status = json.load(r)
    except (urllib.error.URLError, OSError) as exc:
        raise SystemExit(f"No backend at {base} ({exc}). Start it with ./start_backend.sh")
    if not status.get("ready"):
        raise SystemExit(f"Backend at {base} is not ready: {status}")
    print(f"using backend at {base} (device={status.get('device')})\n")

    book, t0 = [], time.time()
    for i, g in enumerate(lib, 1):
        gid = g["gesture_id"]
        print(f"[{i}/{len(lib)}] {gid} {g['intent']}", flush=True)
        payload = json.dumps({
            "prompt": g["kimodo_prompt"], "duration": args.duration,
            "steps": args.steps, "seed": args.seed, "clamp": True,
            "name": gid, "collection": "gesture_library",
        }).encode()
        req = urllib.request.Request(f"{base}/api/generate", data=payload,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=1800) as r:
                d = json.load(r)
        except urllib.error.HTTPError as exc:
            print(f"      FAIL {exc.code}: {exc.read().decode()[:120]}")
            continue
        entry = dict(g)
        entry.update({
            "motion_file": d["csv"], "motion_npz": d["npz"], "motion_video": d["video"],
            "frames": d["frames"], "fps": d["fps"], "duration_s": d["duration_s"],
            "diffusion_steps": d["steps"], "seed": d["seed"],
            "clamped_to_real_robot_limits": d["clamped"],
            "generation_s": d["generation_s"],
            "validation_status": "generated" if d["valid"] else "failed",
            "validation_warnings": [ln.strip() for ln in d["report"].splitlines()
                                    if ln.startswith("WARNING")],
        })
        book.append(entry)
        print(f"      {'OK ' if d['valid'] else 'FAIL'} {d['generation_s']:5.1f}s", flush=True)

    json.dump({"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
               "skeleton": "g1skel34", "model": "Kimodo-G1-RP-v1",
               "source": "backend api", "count": len(book), "gestures": book},
              open(BOOK, "w"), indent=2)
    ok = sum(1 for b in book if b["validation_status"] == "generated")
    print(f"\n{ok}/{len(book)} passed, {time.time() - t0:.0f}s -> {BOOK}")
    return 0 if ok == len(book) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--priority", choices=("High", "Medium", "Low"))
    ap.add_argument("--only", nargs="*", help="Specific gesture_ids")
    ap.add_argument("--duration", type=float, default=3.0)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-encode", action="store_true", help="Reuse an existing embedding cache")
    ap.add_argument("--via-api", metavar="URL", nargs="?", const="http://127.0.0.1:8000",
                    help="Generate through a running backend instead of loading a second "
                         "copy of the models. Avoids the VRAM clash entirely and skips the "
                         "~5 min encode, since the server already holds the encoder.")
    args = ap.parse_args()

    lib = json.load(open(LIB))["gestures"]
    if args.priority:
        lib = [g for g in lib if g["priority"] == args.priority]
    if args.only:
        lib = [g for g in lib if g["gesture_id"] in args.only]
    if not lib:
        print("no gestures selected")
        return 1

    os.makedirs(OUT_ROOT, exist_ok=True)
    print(f"library: {len(lib)} gestures  ->  {OUT_ROOT}\n")

    if args.via_api:
        return build_via_api(lib, args)

    if not args.skip_encode:
        check_vram_free()
        encode_all([g["kimodo_prompt"] for g in lib])

    import mujoco

    from kimodo.assets import skeleton_asset_path
    from kimodo.exports.mujoco import MujocoQposConverter, apply_g1_real_robot_projection
    from kimodo.model.load_model import load_model
    from kimodo.skeleton import global_rots_to_local_rots
    from kimodo.tools import seed_everything

    from generate_g1_motion import PrecomputedTextEncoder
    from render_motion_video import render

    mj = mujoco.MjModel.from_xml_path(str(skeleton_asset_path("g1skel34", "xml", "g1.xml")))
    names = {mj.jnt_qposadr[j]: mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_JOINT, j)
             for j in range(mj.njnt) if mj.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE}

    blob = np.load(EMB, allow_pickle=True)
    table = {str(p): e for p, e in zip(blob["prompts"], blob["embeddings"])}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model("Kimodo-G1-RP-v1", device=device,
                       text_encoder=PrecomputedTextEncoder(table, device=device))
    conv = MujocoQposConverter(model.skeleton.cpu())
    print(f"model ready on {device}\n")

    book, t_start = [], time.time()
    for i, g in enumerate(lib, 1):
        gid, prompt = g["gesture_id"], g["kimodo_prompt"]
        outdir = os.path.join(OUT_ROOT, gid)
        os.makedirs(outdir, exist_ok=True)
        print(f"[{i}/{len(lib)}] {gid} {g['intent']}", flush=True)

        seed_everything(args.seed)
        frames = int(round(args.duration * model.fps))
        t0 = time.time()
        out = model([prompt], frames, constraint_lst=[], num_denoising_steps=args.steps,
                    num_samples=1, multi_prompt=True, num_transition_frames=5,
                    post_processing=False, return_numpy=True)
        gen_s = time.time() - t0

        sk = model.skeleton.cpu()
        pos, rot = apply_g1_real_robot_projection(
            sk, torch.from_numpy(out["posed_joints"][0]),
            torch.from_numpy(out["global_rot_mats"][0]), clamp_to_limits=True)
        qpos = conv.dict_to_qpos(
            {"local_rot_mats": global_rots_to_local_rots(rot, sk)[None],
             "root_positions": pos[:, sk.root_idx, :][None]}, torch.device("cpu"))

        csv_path = os.path.join(outdir, f"{gid}.csv")
        conv.save_csv(qpos, csv_path)
        np.savez(os.path.join(outdir, f"{gid}.npz"),
                 **{k: v for k, v in out.items() if hasattr(v, "shape")})

        proc = subprocess.run([sys.executable, os.path.join(ROOT, "validate_g1_motion.py"), csv_path],
                              capture_output=True, text=True, cwd=ROOT)
        valid = proc.returncode == 0
        warnings = [ln.strip() for ln in proc.stdout.splitlines() if ln.startswith("WARNING")]

        mp4 = os.path.join(outdir, f"{gid}.mp4")
        render(csv_path, mp4, fps=float(model.fps))

        usage, right, left = arm_usage(csv_path, mj, names)
        entry = dict(g)
        entry.update({
            "hand_usage": usage,
            "arm_travel_right": round(right, 2),
            "arm_travel_left": round(left, 2),
            "motion_file": os.path.relpath(csv_path, ROOT),
            "motion_npz": os.path.relpath(os.path.join(outdir, f"{gid}.npz"), ROOT),
            "motion_video": os.path.relpath(mp4, ROOT),
            "frames": frames, "fps": float(model.fps), "duration_s": args.duration,
            "diffusion_steps": args.steps, "seed": args.seed,
            "clamped_to_real_robot_limits": True,
            "generation_s": round(gen_s, 2),
            "validation_status": "generated" if valid else "failed",
            "validation_warnings": warnings,
        })
        book.append(entry)
        flag = "OK " if valid else "FAIL"
        print(f"      {flag} {gen_s:5.1f}s  hand_usage={usage:16s} R/L={right:5.2f}/{left:5.2f}"
              + (f"  [{len(warnings)} warning(s)]" if warnings else ""), flush=True)

    json.dump({"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
               "skeleton": "g1skel34", "model": "Kimodo-G1-RP-v1",
               "count": len(book), "gestures": book},
              open(BOOK, "w"), indent=2)

    ok = sum(1 for b in book if b["validation_status"] == "generated")
    warned = sum(1 for b in book if b["validation_warnings"])
    print(f"\n{'=' * 62}")
    print(f"{ok}/{len(book)} passed validation, {warned} with warnings, "
          f"{time.time() - t_start:.0f}s total")
    print(f"gesture book: {BOOK}")
    print(f"motions     : {OUT_ROOT}/<gesture_id>/")
    return 0 if ok == len(book) else 1


if __name__ == "__main__":
    sys.exit(main())
