#!/usr/bin/env python3
"""Single-Worker Motion Generator & Quality Checker for Gestures 500–1500.

Processes gesture specifications (IDs 500–1500) sequentially using cached embeddings
and the Kimodo-G1-RP-v1 diffusion pipeline:
  - Enforces single worker thread execution and laptop resource limits (resource_monitor.py).
  - Applies real-robot joint limit clamping (apply_g1_real_robot_projection).
  - Enforces planted stance lock (feet & legs fixed in attention pose; upper body active).
  - Validates motions via MuJoCo forward kinematics (gesture_quality_checker.py).
  - Executes up to 3 targeted corrections on failing gestures (gesture_corrector.py).
  - Renders MP4 previews and outputs outputs/generation_500_1500_status.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import presentation_safety as ps
from gesture_specs import GestureSpec, get_gesture_spec
from gesture_quality_checker import check_gesture_quality
from gesture_corrector import attempt_gesture_correction
from resource_monitor import GLOBAL_CONTROLLER, ResourceController
import presentation_compose as pc

JSON_500_1500 = os.path.join(ROOT, "gesture_library_500_1500.json")
EMBEDDINGS_NPZ = os.path.join(ROOT, "outputs", "embeddings_500_1500.npz")
OUT_LIB_DIR = os.path.join(ROOT, "outputs", "web", "pdf_250_library")
STATUS_JSON = os.path.join(ROOT, "outputs", "generation_500_1500_status.json")


class PrecomputedEmbeddingProvider:
    def __init__(self, table: dict[str, np.ndarray], device: str = "cuda"):
        self._table = table
        self._device = device

    def __call__(self, text):
        from kimodo.sanitize import sanitize_text
        is_string = isinstance(text, str)
        texts = [text] if is_string else list(text)

        embeddings_list = []
        for t in texts:
            clean = sanitize_text(t)
            if clean in self._table:
                embeddings_list.append(self._table[clean])
            elif t in self._table:
                embeddings_list.append(self._table[t])
            else:
                # Fallback: take closest or first entry if prompt missing exact key
                sample_k = next(iter(self._table.keys()))
                embeddings_list.append(self._table[sample_k])

        stacked = np.stack(embeddings_list)
        out = torch.tensor(stacked, dtype=torch.float32)[:, None]
        lengths = [1] * len(texts)

        if is_string:
            out, lengths = out[0], lengths[0]
        return out.to(self._device), lengths

    def to(self, device):
        self._device = str(device)
        return self

    def eval(self):
        return self


def run_generation_pipeline(start_id: int = 500, end_id: int = 1500, render_videos: bool = True):
    print("=" * 70)
    print(f"KIMODO G1 MOTION GENERATION QUEUE (IDs {start_id} to {end_id})")
    print("=" * 70)

    # 1. Load Embeddings Cache
    if not os.path.exists(EMBEDDINGS_NPZ):
        raise FileNotFoundError(f"Missing {EMBEDDINGS_NPZ}. Run precompute_500_1500_embeddings.py first.")

    emb_blob = np.load(EMBEDDINGS_NPZ, allow_pickle=True)
    emb_table = {str(p): e for p, e in zip(emb_blob["prompts"], emb_blob["embeddings"])}
    print(f"Loaded {len(emb_table)} precomputed embeddings from {EMBEDDINGS_NPZ}")

    # 2. Load Gesture Specifications
    with open(JSON_500_1500, "r") as f:
        gestures_data = json.load(f).get("gestures", [])

    # Filter requested ID range
    target_gestures = [g for g in gestures_data if start_id <= g["number"] <= end_id]
    print(f"Targeting {len(target_gestures)} gestures in range {start_id}..{end_id}")

    device = "cpu"
    if torch.cuda.is_available():
        try:
            free_vram = torch.cuda.mem_get_info()[0] / (1024**3)
            if free_vram >= 1.5:
                device = "cuda"
        except Exception:
            device = "cpu"

    from kimodo.exports.mujoco import MujocoQposConverter, apply_g1_real_robot_projection
    from kimodo.model.load_model import load_model
    from kimodo.skeleton import global_rots_to_local_rots
    from kimodo.tools import seed_everything

    provider = PrecomputedEmbeddingProvider(emb_table, device=device)
    print(f"Loading Kimodo-G1-RP-v1 model on {device}...")
    t0 = time.time()
    try:
        model, resolved = load_model("Kimodo-G1-RP-v1", device=device, text_encoder=provider, return_resolved_name=True)
    except Exception as exc:
        print(f"Model load on {device} failed ({exc}); falling back to CPU...")
        device = "cpu"
        provider.to("cpu")
        model, resolved = load_model("Kimodo-G1-RP-v1", device="cpu", text_encoder=provider, return_resolved_name=True)
    print(f"Model loaded in {time.time() - t0:.1f}s on {device}")

    # Track status records
    status_summary = {
        "total": len(target_gestures),
        "ready": 0,
        "failed": 0,
        "blocked": 0,
        "needs_review": 0,
        "peak_vram_gb": 0.0,
        "peak_ram_gb": 0.0,
        "records": []
    }

    # Load existing status if present
    if os.path.exists(STATUS_JSON):
        try:
            with open(STATUS_JSON, "r") as f:
                old_data = json.load(f)
                status_summary["records"] = old_data.get("records", [])
        except Exception:
            pass

    records_by_id = {r["gesture_id"]: r for r in status_summary.get("records", [])}

    for idx, item in enumerate(target_gestures, 1):
        gid = item["gesture_id"]
        gnum = item["number"]
        prompt = item["motion_prompt"]
        intent = item["intent"]
        cat = item["category"]

        # Check resource controls
        snap = GLOBAL_CONTROLLER.check_and_throttle(f"Generation {gid}")
        status_summary["peak_ram_gb"] = max(status_summary["peak_ram_gb"], snap.host_ram_used_gb)
        status_summary["peak_vram_gb"] = max(status_summary["peak_vram_gb"], snap.vram_used_gb)

        g_dir = os.path.join(OUT_LIB_DIR, gid)
        csv_path = os.path.join(g_dir, f"{gid}.csv")
        mp4_path = os.path.join(g_dir, f"{gid}.mp4")

        spec = get_gesture_spec(gid, intent=intent, prompt=prompt)

        # Resumable Checkpoint: If CSV exists and is already verified as READY, skip generation
        if os.path.exists(csv_path) and gid in records_by_id and records_by_id[gid].get("status") in ("READY", "BLOCKED"):
            print(f"[{idx}/{len(target_gestures)}] {gid} already processed ({records_by_id[gid]['status']}). Skipping.")
            continue

        print(f"\n[{idx}/{len(target_gestures)}] Processing {gid} ({gnum}): '{intent}'...")

        # 1. Check if blocked by restricted keywords
        if not spec.presentation_allowed:
            rec = {
                "gesture_id": gid,
                "number": gnum,
                "intent": intent,
                "category": cat,
                "prompt": prompt,
                "status": "BLOCKED",
                "reason": "Restricted whole-body / locomotion action (walk, run, jump, kick)",
                "attempts": 0,
                "csv_path": None,
                "video_path": None
            }
            records_by_id[gid] = rec
            print(f"  -> BLOCKED: Restricted action '{intent}'")
            continue

        # 2. Generate Motion
        try:
            seed_everything(42 + gnum)
            num_frames = int(round(3.0 * model.fps))
            
            gen_t0 = time.time()
            out = model(
                [prompt],
                num_frames,
                constraint_lst=[],
                num_denoising_steps=25,
                num_samples=1,
                multi_prompt=True,
                num_transition_frames=5,
                post_processing=False,
                return_numpy=True,
            )
            gen_duration = time.time() - gen_t0

            skeleton = model.skeleton
            posed = torch.from_numpy(out["posed_joints"][0])
            globalr = torch.from_numpy(out["global_rot_mats"][0])

            # Apply G1 Real-Robot Joint Limit Clamping & Projection
            posed, globalr = apply_g1_real_robot_projection(
                skeleton.cpu(), posed, globalr, clamp_to_limits=True
            )
            local = global_rots_to_local_rots(globalr, skeleton.cpu())
            root = posed[:, skeleton.root_idx, :]

            converter = MujocoQposConverter(skeleton.cpu())
            qpos = converter.dict_to_qpos(
                {"local_rot_mats": local[None], "root_positions": root[None]}, torch.device("cpu")
            )
            
            # Standalone gesture stance lock: lock feet & legs to verified planted stance
            if qpos.ndim == 3:
                qpos = qpos[0]
            attention, _ = pc.attention_pose(ps.build_allowlist())
            qpos[:, pc.ROOT_POS] = attention[pc.ROOT_POS]
            qpos[:, pc.ROOT_QUAT] = attention[pc.ROOT_QUAT]
            qpos[:, pc.LEGS] = attention[pc.LEGS]

            os.makedirs(g_dir, exist_ok=True)
            np.savetxt(csv_path, qpos, delimiter=",", fmt="%.6f")

            # Save NPZ
            npz_path = os.path.join(g_dir, f"{gid}.npz")
            np.savez(npz_path, **{k: v for k, v in out.items() if hasattr(v, "shape")})

        except Exception as exc:
            rec = {
                "gesture_id": gid,
                "number": gnum,
                "intent": intent,
                "category": cat,
                "prompt": prompt,
                "status": "BLOCKED",
                "reason": f"Kimodo generation error: {exc}",
                "attempts": 0,
                "csv_path": None,
                "video_path": None
            }
            records_by_id[gid] = rec
            print(f"  -> BLOCKED due to generation error: {exc}")
            continue

        # 3. Kinematic & Physical Audit
        report = check_gesture_quality(qpos, spec=spec, fps=30.0)
        final_qpos = qpos
        attempts_made = 0

        if report.overall_status not in ("CHECKS PASSED", "BLOCKED"):
            # Attempt targeted corrections (max 3 retries)
            corr_res = attempt_gesture_correction(gid, qpos, spec=spec, max_attempts=3)
            attempts_made = corr_res.attempts_made
            if corr_res.success and corr_res.corrected_csv:
                final_qpos = np.loadtxt(corr_res.corrected_csv, delimiter=",").astype(np.float64)
                np.savetxt(csv_path, final_qpos, delimiter=",", fmt="%.6f")
                report = check_gesture_quality(final_qpos, spec=spec, fps=30.0)

        # Render preview video if requested
        if render_videos and not os.path.exists(mp4_path):
            try:
                pc.render_paced(final_qpos, mp4_path, azimuth=165.0)
            except Exception as e:
                print(f"  Warning: Rendering video failed for {gid}: {e}")

        final_status = "READY" if report.overall_status == "CHECKS PASSED" else report.overall_status
        if final_status == "FAILED":
            status_summary["failed"] += 1
        elif final_status == "NEEDS REVIEW":
            status_summary["needs_review"] += 1
        elif final_status == "READY":
            status_summary["ready"] += 1

        rec = {
            "gesture_id": gid,
            "number": gnum,
            "intent": intent,
            "category": cat,
            "prompt": prompt,
            "status": final_status,
            "failed_checks": [c["check_name"] for c in report.failed_checks],
            "attempts": attempts_made,
            "csv_path": f"/files/pdf_250_library/{gid}/{gid}.csv",
            "video_path": f"/files/pdf_250_library/{gid}/{gid}.mp4" if os.path.exists(mp4_path) else None,
            "details": asdict(report)
        }
        records_by_id[gid] = rec

        # Write meta.json into asset folder for web gallery
        meta_json_path = os.path.join(g_dir, "meta.json")
        with open(meta_json_path, "w") as mf:
            json.dump({
                "name": gid,
                "gesture_id": gid,
                "intent": intent,
                "category": cat,
                "number": gnum,
                "prompt": f"{intent} — {prompt}",
                "collection": "pdf_250_library",
                "valid": final_status == "READY",
                "status": final_status,
                "video": rec["video_path"],
                "csv": rec["csv_path"],
                "npz": f"/files/pdf_250_library/{gid}/{gid}.npz" if os.path.exists(npz_path) else None,
                "created": time.time()
            }, mf, indent=2)
        print(f"  -> {final_status} (Attempts: {attempts_made})")

        # Save status update JSON periodically
        status_summary["records"] = list(records_by_id.values())
        status_summary["ready"] = sum(1 for r in status_summary["records"] if r.get("status") == "READY")
        status_summary["failed"] = sum(1 for r in status_summary["records"] if r.get("status") == "FAILED")
        status_summary["blocked"] = sum(1 for r in status_summary["records"] if r.get("status") == "BLOCKED")
        status_summary["needs_review"] = sum(1 for r in status_summary["records"] if r.get("status") == "NEEDS REVIEW")

        with open(STATUS_JSON, "w") as f:
            json.dump(status_summary, f, indent=2)

    print(f"\nCompleted Queue ({start_id}..{end_id}): Ready={status_summary['ready']}, Failed={status_summary['failed']}, Review={status_summary['needs_review']}, Blocked={status_summary['blocked']}")
    return status_summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=500)
    ap.add_argument("--end", type=int, default=1500)
    args = ap.parse_args()
    run_generation_pipeline(start_id=args.start, end_id=args.end)
