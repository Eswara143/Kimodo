#!/usr/bin/env python3
"""Master Re-Training & Iterative Motion Refinement Pipeline for Unitree G1 (Gestures 500-1500).

Performs strict, end-to-end iterative quality validation and targeted kinematic refinement
for all 1,001 library gestures based on their authoritative stored instructions.
Preserves existing IDs, spoken phrases, and intents.
Tracks attempt history and maintains persistent JSON state checkpoints for seamless resume.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import presentation_safety as ps
from gesture_specs import GestureSpec, get_gesture_spec
from gesture_quality_checker import check_gesture_quality, QualityReport
from gesture_corrector import (
    attempt_gesture_correction,
    _swap_arm_channels,
    _mirror_arm_channels,
    _scale_arm_amplitude,
    _smooth_joint_jumps,
    _extend_peak_hold,
    _resolve_self_collision,
    _dampen_velocity_acceleration,
    _synthesize_clean_arm_action
)
from render_motion_video import render_video

CATALOGUE_PATH = os.path.join(ROOT, "gesture_library_500_1500.json")
LIBRARY_DIR = os.path.join(ROOT, "outputs", "web", "pdf_250_library")
ATTEMPT_HISTORY_DIR = os.path.join(ROOT, "outputs", "attempt_history")
CHECKPOINT_PATH = os.path.join(ROOT, "outputs", "master_retrain_checkpoint.json")


def load_catalogue() -> list[dict]:
    if not os.path.exists(CATALOGUE_PATH):
        raise FileNotFoundError(f"Catalogue JSON not found at {CATALOGUE_PATH}")
    with open(CATALOGUE_PATH, "r") as f:
        data = json.load(f)
    return data.get("gestures", [])


def load_checkpoint() -> dict:
    if os.path.exists(CHECKPOINT_PATH):
        try:
            with open(CHECKPOINT_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "last_processed_id": None,
        "total_target": 1001,
        "total_checked": 0,
        "total_retrained": 0,
        "total_passed_first_attempt": 0,
        "total_required_correction": 0,
        "total_iterations": 0,
        "total_ready": 0,
        "total_failed": 0,
        "total_blocked": 0,
        "total_needs_review": 0,
        "most_iterations_gesture": None,
        "max_iterations_count": 0,
        "gesture_status": {},
    }


def save_checkpoint(ckpt: dict) -> None:
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(ckpt, f, indent=2)


def process_gesture_refinement(item: dict, max_attempts: int = 5, fps: float = 30.0) -> dict:
    num = item.get("number", 0)
    gid = item.get("gesture_id", f"PDF_{num:03d}")
    phrase = item.get("spoken_response") or item.get("intent") or ""
    action = item.get("source_action") or item.get("motion_prompt") or ""
    intent = item.get("intent") or phrase

    spec = get_gesture_spec(gid, intent=intent, prompt=action)
    g_dir = os.path.join(LIBRARY_DIR, gid)
    os.makedirs(g_dir, exist_ok=True)

    csv_path = os.path.join(g_dir, f"{gid}.csv")
    npz_path = os.path.join(g_dir, f"{gid}.npz")
    mp4_path = os.path.join(g_dir, f"{gid}.mp4")
    meta_path = os.path.join(g_dir, "meta.json")

    att_dir = os.path.join(ATTEMPT_HISTORY_DIR, gid)
    os.makedirs(att_dir, exist_ok=True)

    # 1. Load existing motion trajectory or synthesize baseline qpos
    qpos = None
    if os.path.exists(csv_path):
        try:
            qpos = np.loadtxt(csv_path, delimiter=",")
        except Exception:
            pass

    if qpos is None or len(qpos) < 5:
        # Generate baseline pose sequence from default attention posture
        att_pose = np.zeros(36)
        n_frames = 90
        qpos = np.tile(att_pose, (n_frames, 1))

    # Initial quality evaluation
    report = check_gesture_quality(qpos, spec=spec, fps=fps)
    attempt_history: list[dict] = []
    attempts = 0
    accepted = report.overall_status == "CHECKS PASSED"

    if accepted:
        attempts = 1
        att_history_entry = {
            "attempt_number": 1,
            "strategy": "BASELINE_VERIFIED",
            "description": "Existing baseline gesture motion passed all technical and semantic quality checks.",
            "passed": True,
            "remaining_failures": [],
            "status": "CHECKS PASSED"
        }
        attempt_history.append(att_history_entry)
        np.savetxt(os.path.join(att_dir, "attempt_1.csv"), qpos, delimiter=",", fmt="%.6f")
        with open(os.path.join(att_dir, "attempt_1_report.json"), "w") as f:
            json.dump(asdict(report), f, indent=2)
    else:
        # Iterative Refinement Loop
        current_qpos = qpos.copy()
        current_report = report

        for att in range(1, max_attempts + 1):
            attempts = att
            failed_names = [c["check_name"] for c in current_report.failed_checks]
            cand_qpos = current_qpos.copy()
            strategy = ""
            desc = ""

            # Failure diagnosis & strategy selection
            if att >= 3 and ("Correct Arm Execution" in failed_names or "Time-Based Joint Continuity" in failed_names or "Gesture Hold Duration" in failed_names):
                strategy = "SYNTHESIZE_CLEAN_ARM_ACTION"
                desc = f"Synthesized clean, kinematically smooth presentation trajectory for target {spec.arm.upper()} arm."
                cand_qpos = _synthesize_clean_arm_action(cand_qpos, target_arm=spec.arm, amplitude=0.45)
                cand_qpos = _extend_peak_hold(cand_qpos, hold_frames=int(spec.min_hold_seconds * fps))

            elif "Self-Collision & Clearance" in failed_names:
                strategy = "RESOLVE_SELF_COLLISION"
                offset = 0.22 if att >= 2 else 0.14
                desc = f"Adjusted arm shoulder roll/yaw and elbow joints outward by {offset:.2f} rad to eliminate self-collision."
                cand_qpos = _resolve_self_collision(cand_qpos, offset_rad=offset)

            elif "Joint Acceleration Ceiling" in failed_names or "Joint Velocity Ceiling" in failed_names:
                strategy = "DAMPEN_VELOCITY_ACCELERATION"
                desc = "Interpolated frames temporally (1.3x) and applied 3-point smoothing to reduce velocity & acceleration."
                cand_qpos = _dampen_velocity_acceleration(cand_qpos, factor=1.3)

            elif "Correct Arm Execution" in failed_names:
                if spec.arm == "both":
                    strategy = "MIRROR_ARM_CHANNELS"
                    desc = "Mirrored active arm excursions to both left and right arm channels."
                    cand_qpos = _mirror_arm_channels(cand_qpos)
                else:
                    strategy = "SWAP_ARM_CHANNELS"
                    desc = f"Swapped left/right arm channels with reflection for target {spec.arm.upper()} arm execution."
                    cand_qpos = _swap_arm_channels(cand_qpos)

            elif "Wave Repetition Count" in failed_names:
                strategy = "SCALE_ARM_AMPLITUDE"
                desc = f"Amplified arm range of motion by 1.4x for {spec.arm} arm to ensure visible execution."
                cand_qpos = _scale_arm_amplitude(cand_qpos, spec.arm, scale=1.40)

            elif "Time-Based Joint Continuity" in failed_names or "Time-Based Joint Continuity (No Jumps)" in failed_names:
                strategy = "SMOOTH_JOINT_JUMPS"
                desc = "Applied moving average trajectory smoothing to eliminate step jumps."
                cand_qpos = _smooth_joint_jumps(cand_qpos)

            elif "Gesture Hold Duration" in failed_names:
                strategy = "EXTEND_PEAK_HOLD"
                desc = f"Inserted peak hold frames for {spec.min_hold_seconds:.2f}s static hold at peak posture."
                cand_qpos = _extend_peak_hold(cand_qpos, hold_frames=int(spec.min_hold_seconds * fps))

            elif "Stationary Feet & Base Stability" in failed_names or "Hand-Only Torso & Lower-Body Stillness" in failed_names:
                strategy = "STATIONARY_BODY_CLAMP"
                desc = "Locked pelvic, torso, and leg joints to attention baseline to enforce stationary body constraints."
                att_pose = np.zeros(36)
                cand_qpos[:, :22] = att_pose[:22]

            else:
                strategy = "ARM_AMPLITUDE_MODULATION"
                desc = f"Modulated arm trajectory scaling (1.25x) for {spec.arm} arm."
                cand_qpos = _scale_arm_amplitude(cand_qpos, spec.arm, scale=1.25)

            # Evaluate candidate motion
            cand_report = check_gesture_quality(cand_qpos, spec=spec, fps=fps)
            passed = cand_report.overall_status == "CHECKS PASSED"
            rem_fails = [c["check_name"] for c in cand_report.failed_checks]

            att_history_entry = {
                "attempt_number": att,
                "strategy": strategy,
                "description": desc,
                "passed": passed,
                "remaining_failures": rem_fails,
                "status": cand_report.overall_status
            }
            attempt_history.append(att_history_entry)

            # Save attempt files
            np.savetxt(os.path.join(att_dir, f"attempt_{att}.csv"), cand_qpos, delimiter=",", fmt="%.6f")
            with open(os.path.join(att_dir, f"attempt_{att}_report.json"), "w") as f:
                json.dump(asdict(cand_report), f, indent=2)

            if passed:
                accepted = True
                current_qpos = cand_qpos
                current_report = cand_report
                break
            else:
                # If candidate improved overall status or reduced failures, retain it
                if len(cand_report.failed_checks) <= len(current_report.failed_checks):
                    current_qpos = cand_qpos
                    current_report = cand_report

        qpos = current_qpos
        report = current_report

    # Save FINAL_ACCEPTED or latest refined motion
    final_status = "READY" if accepted else (report.overall_status if report.overall_status in ("BLOCKED", "NEEDS REVIEW") else "FAILED")
    np.savetxt(csv_path, qpos, delimiter=",", fmt="%.6f")

    # Render MP4 video preview if missing or updated
    if not os.path.exists(mp4_path) or os.path.getsize(mp4_path) < 1000:
        try:
            render_video(qpos, mp4_path, title=f"{gid}: {phrase}")
        except Exception as e:
            print(f"[{gid}] MP4 render warning: {e}")

    # Generate metadata JSON
    meta = {
        "gesture_id": gid,
        "number": num,
        "spoken_response": phrase,
        "intent": intent,
        "source_action": action,
        "category": item.get("category", "Dialogue & Technical Catalogue 500-1500"),
        "status": final_status,
        "accepted_version": "FINAL_ACCEPTED" if accepted else "REFINED_CANDIDATE",
        "attempts_made": attempts,
        "attempt_history": attempt_history,
        "quality_report": asdict(report)
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return {
        "gesture_id": gid,
        "status": final_status,
        "accepted": accepted,
        "attempts": attempts,
        "attempt_history": attempt_history,
        "report": report
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=int, default=500, help="Starting gesture number (default: 500)")
    ap.add_argument("--end", type=int, default=1500, help="Ending gesture number (default: 1500)")
    ap.add_argument("--max-attempts", type=int, default=5, help="Max refinement attempts per gesture")
    args = ap.parse_args()

    catalogue = load_catalogue()
    target_items = [g for g in catalogue if args.start <= g.get("number", 0) <= args.end]
    print(f"=== Master Re-Training & Refinement Engine ===")
    print(f"Target range: IDs {args.start} to {args.end} ({len(target_items)} gestures)")

    ckpt = load_checkpoint()
    t_start = time.time()

    for idx, item in enumerate(target_items, start=1):
        num = item.get("number", 0)
        gid = item.get("gesture_id", f"PDF_{num:03d}")

        res = process_gesture_refinement(item, max_attempts=args.max_attempts)
        status = res["status"]

        # Update checkpoint metrics
        ckpt["last_processed_id"] = gid
        ckpt["total_checked"] += 1
        if res["attempts"] == 1 and res["accepted"]:
            ckpt["total_passed_first_attempt"] += 1
        else:
            ckpt["total_required_correction"] += 1
            ckpt["total_retrained"] += 1

        ckpt["total_iterations"] += res["attempts"]
        if res["attempts"] > ckpt["max_iterations_count"]:
            ckpt["max_iterations_count"] = res["attempts"]
            ckpt["most_iterations_gesture"] = gid

        ckpt["gesture_status"][gid] = {
            "status": status,
            "attempts": res["attempts"],
            "accepted": res["accepted"],
            "phrase": item.get("spoken_response", "")
        }

        # Count overall tally
        all_st = [v["status"] for v in ckpt["gesture_status"].values()]
        ckpt["total_ready"] = all_st.count("READY")
        ckpt["total_failed"] = all_st.count("FAILED")
        ckpt["total_blocked"] = all_st.count("BLOCKED")
        ckpt["total_needs_review"] = all_st.count("NEEDS REVIEW") + all_st.count("NEEDS_REVIEW")

        save_checkpoint(ckpt)

        if idx % 50 == 0 or idx == len(target_items):
            print(f"Progress [{idx}/{len(target_items)}] ({gid}): status={status}, attempts={res['attempts']} | Ready: {ckpt['total_ready']}, Failed: {ckpt['total_failed']}")

    elapsed_s = time.time() - t_start
    print("\n=== Master Refinement Run Completed ===")
    print(f"Total gestures checked: {ckpt['total_checked']}")
    print(f"Passed 1st attempt    : {ckpt['total_passed_first_attempt']}")
    print(f"Required correction   : {ckpt['total_required_correction']}")
    print(f"Total iterations      : {ckpt['total_iterations']}")
    print(f"TOTAL READY           : {ckpt['total_ready']}")
    print(f"TOTAL FAILED          : {ckpt['total_failed']}")
    print(f"TOTAL BLOCKED         : {ckpt['total_blocked']}")
    print(f"TOTAL NEEDS REVIEW    : {ckpt['total_needs_review']}")
    print(f"Elapsed time          : {elapsed_s:.1f}s")


if __name__ == "__main__":
    main()
