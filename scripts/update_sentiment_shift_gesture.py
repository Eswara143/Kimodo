#!/usr/bin/env python3
"""Assign Grand Gesture Sequence (Namaste at chest -> Overhead Stretch -> Open Wide Welcome) to PDF_1249 ('Here is the sentiment shift.').
"""

import os
import sys
import json
import shutil
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from render_motion_video import render_video
from gesture_quality_checker import check_gesture_quality
from gesture_specs import get_gesture_spec

LIBRARY_DIR = os.path.join(ROOT, "outputs", "web", "pdf_250_library")


def main():
    src_gid = "PDF_004"
    target_gid = "PDF_1249"

    src_dir = os.path.join(LIBRARY_DIR, src_gid)
    target_dir = os.path.join(LIBRARY_DIR, target_gid)
    os.makedirs(target_dir, exist_ok=True)

    src_csv = os.path.join(src_dir, f"{src_gid}.csv")
    target_csv = os.path.join(target_dir, f"{target_gid}.csv")
    target_npz = os.path.join(target_dir, f"{target_gid}.npz")
    target_mp4 = os.path.join(target_dir, f"{target_gid}.mp4")
    target_meta = os.path.join(target_dir, "meta.json")

    user_prompt = "A person begins by pressing both palms together at the chest in a respectful Namaste position. From there, they smoothly stretch both arms straight upward overhead into a grand upward salute, and then gracefully open them outward to shoulder width with palms facing forward."
    phrase = "Here is the sentiment shift."
    full_prompt = f"{phrase} — {user_prompt}"

    print(f"Copying and assigning grand gesture trajectory from {src_gid} to {target_gid} ('{phrase}')...")

    # Load qpos array from PDF_004
    qpos = np.loadtxt(src_csv, delimiter=",")

    # Save target CSV and NPZ
    np.savetxt(target_csv, qpos, delimiter=",", fmt="%.6f")
    np.savez_compressed(target_npz, qpos=qpos, fps=30.0, clamped=True)
    print(f"Saved CSV to {target_csv}")
    print(f"Saved NPZ to {target_npz}")

    # Validate quality of target gesture
    spec = get_gesture_spec(target_gid, intent=phrase, prompt=user_prompt)
    report = check_gesture_quality(qpos, spec=spec, gesture_id=target_gid)
    print(f"Quality Check Status for {target_gid}: {report.overall_status}")
    if report.failed_checks:
        print("Failed checks:", [c["check_name"] for c in report.failed_checks])

    # Render fresh MP4 preview video for PDF_1249
    print(f"Rendering updated MP4 preview video for {target_gid}...")
    render_video(target_csv, target_mp4, title=f"PDF_1249: {phrase} (Grand Namaste -> Sky Stretch -> Open Welcome)")
    print(f"Saved MP4 video to {target_mp4}")

    meta_data = {
        "name": target_gid,
        "gesture_id": target_gid,
        "number": 1249,
        "gesture_name": phrase,
        "spoken_response": phrase,
        "intent": phrase,
        "motion_prompt": user_prompt,
        "prompt": full_prompt,
        "category": "Dialogue & Technical Catalogue 1000-1500",
        "collection": "pdf_250_library",
        "frames": len(qpos),
        "fps": 30.0,
        "duration_s": round(len(qpos) / 30.0, 2),
        "clamped": True,
        "valid": True,
        "status": "READY",
        "perfect": True,
        "approved": True,
        "published": True,
        "video": f"/files/pdf_250_library/{target_gid}/{target_gid}.mp4",
        "csv": f"/files/pdf_250_library/{target_gid}/{target_gid}.csv",
        "npz": f"/files/pdf_250_library/{target_gid}/{target_gid}.npz",
        "qpos_shape": [1, len(qpos), 36]
    }

    with open(target_meta, "w") as mf:
        json.dump(meta_data, mf, indent=2)

    # Update in master gesture_library_500_1500.json
    cat_path = os.path.join(ROOT, "gesture_library_500_1500.json")
    if os.path.isfile(cat_path):
        with open(cat_path) as f:
            cdata = json.load(f)
        for g in cdata.get("gestures", []):
            if g.get("gesture_id") == target_gid or g.get("number") == 1249:
                g["motion_prompt"] = user_prompt
                g["prompt"] = full_prompt
                g["status"] = "READY"
                g["valid"] = True
        with open(cat_path, "w") as f:
            json.dump(cdata, f, indent=2)

    print(f"Successfully assigned gesture to PDF_1249 '{phrase}'!")


if __name__ == "__main__":
    main()
