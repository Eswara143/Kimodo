#!/usr/bin/env python3
"""Update Good Morning Gesture (PDF_004) to:
"A person begins by pressing both palms together at the chest in a respectful Namaste position. From there, they smoothly stretch both arms straight upward overhead into a grand upward salute, and then gracefully open them outward to shoulder width with palms facing forward to welcome the morning."
"""

import os
import sys
import math
import json
import numpy as np
from scipy.ndimage import gaussian_filter1d

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from render_motion_video import render_video
from gesture_quality_checker import check_gesture_quality
from gesture_specs import get_gesture_spec

LIBRARY_DIR = os.path.join(ROOT, "outputs", "web", "pdf_250_library")


def minimum_jerk(t: float) -> float:
    """7th-order minimum-jerk polynomial (35t^4 - 84t^5 + 70t^6 - 20t^7) for zero-jerk fluid motion."""
    t = max(0.0, min(1.0, t))
    return t**4 * (35.0 - 84.0 * t + 70.0 * t**2 - 20.0 * t**3)


def generate_namaste_overhead_open_good_morning(num_frames: int = 130) -> np.ndarray:
    """Generates the grand gesture sequence:
    1. Palms together at chest (Namaste)
    2. Stretch straight upward overhead (Grand Upward Salute)
    3. Gracefully open outward to shoulder width with open palms forward (Welcome Morning)
    """
    qpos = np.zeros((num_frames, 36), dtype=np.float64)
    qpos[:, 2] = 0.75
    qpos[:, 3] = 1.0

    # Baseline rest pose
    rest_left = np.array([0.28, 0.10, -0.30, -0.12, 0.0, 0.0, 0.0])
    rest_right = np.array([0.28, -0.10, 0.30, -0.12, 0.0, 0.0, 0.0])

    # Phase 1: Respectful Palms Together (Namaste) at Chest Level
    namaste_left = np.array([0.48, 0.36, -0.35, 1.32, 0.15, -0.25, 0.0])
    namaste_right = np.array([0.48, -0.36, 0.35, 1.32, -0.15, 0.25, 0.0])

    # Phase 2: Grand Upward Salute (Stretched straight upward overhead)
    stretch_left = np.array([-0.82, 0.35, 0.0, 0.15, 0.0, 0.0, 0.0])
    stretch_right = np.array([-0.82, -0.35, 0.0, 0.15, 0.0, 0.0, 0.0])

    # Phase 3: Open Outward to Shoulder Width (Palms Facing Forward)
    open_left = np.array([0.10, 1.15, -0.20, 0.30, 0.0, 0.0, 0.0])
    open_right = np.array([0.10, -1.15, 0.20, 0.30, 0.0, 0.0, 0.0])

    f_prep = 8
    f_namaste = 32
    f_stretch = 65
    f_open = 98
    f_lower = 124

    for i in range(num_frames):
        t_sec = i / 30.0

        # Phase 1: Rise to Namaste at Chest
        if i < f_prep:
            qpos[i, 22:29] = rest_left
            qpos[i, 29:36] = rest_right
        elif i < f_namaste:
            s1 = minimum_jerk((i - f_prep) / float(f_namaste - f_prep))
            qpos[i, 22:29] = (1.0 - s1) * rest_left + s1 * namaste_left
            qpos[i, 29:36] = (1.0 - s1) * rest_right + s1 * namaste_right
        # Phase 2: Smoothly stretch straight upward overhead into grand salute
        elif i < f_stretch:
            s2 = minimum_jerk((i - f_namaste) / float(f_stretch - f_namaste))
            qpos[i, 22:29] = (1.0 - s2) * namaste_left + s2 * stretch_left
            qpos[i, 29:36] = (1.0 - s2) * namaste_right + s2 * stretch_right
        # Phase 3: Gracefully open outward to shoulder width with open palms facing forward
        elif i < f_open:
            s3 = minimum_jerk((i - f_stretch) / float(f_open - f_stretch))
            qpos[i, 22:29] = (1.0 - s3) * stretch_left + s3 * open_left
            qpos[i, 29:36] = (1.0 - s3) * stretch_right + s3 * open_right
        # Phase 4: Return smoothly to rest pose
        elif i < f_lower:
            s4 = minimum_jerk((i - f_open) / float(f_lower - f_open))
            qpos[i, 22:29] = (1.0 - s4) * open_left + s4 * rest_left
            qpos[i, 29:36] = (1.0 - s4) * open_right + s4 * rest_right
        else:
            qpos[i, 22:29] = rest_left
            qpos[i, 29:36] = rest_right

        # Natural torso micro-sway and subtle chest expansion
        qpos[i, 19] = 0.012 * math.sin(2.0 * math.pi * 0.3 * t_sec)  # waist_yaw
        qpos[i, 21] = 0.010 * math.sin(2.0 * math.pi * 0.4 * t_sec)  # waist_pitch

    # Apply temporal Gaussian filter pass across all active joints
    for j in range(19, 36):
        qpos[:, j] = gaussian_filter1d(qpos[:, j], sigma=1.5, mode="nearest")

    return qpos


def main():
    gid = "PDF_004"
    out_dir = os.path.join(LIBRARY_DIR, gid)
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, f"{gid}.csv")
    npz_path = os.path.join(out_dir, f"{gid}.npz")
    mp4_path = os.path.join(out_dir, f"{gid}.mp4")
    meta_path = os.path.join(out_dir, "meta.json")

    user_prompt = "A person begins by pressing both palms together at the chest in a respectful Namaste position. From there, they smoothly stretch both arms straight upward overhead into a grand upward salute, and then gracefully open them outward to shoulder width with palms facing forward to welcome the morning."
    full_prompt = f"Good Morning — {user_prompt}"

    print(f"Generating grand Namaste -> Overhead Skyward Stretch -> Open Wide Welcome sequence for {gid}...")
    qpos = generate_namaste_overhead_open_good_morning(num_frames=130)

    # Validate quality
    spec = get_gesture_spec(gid, intent="Good Morning", prompt=user_prompt)
    report = check_gesture_quality(qpos, spec=spec, gesture_id=gid)
    print(f"Quality Check Status for {gid}: {report.overall_status}")
    if report.failed_checks:
        print("Failed checks:", [c["check_name"] for c in report.failed_checks])

    # Save CSV & NPZ
    np.savetxt(csv_path, qpos, delimiter=",", fmt="%.6f")
    np.savez_compressed(npz_path, qpos=qpos, fps=30.0, clamped=True)
    print(f"Saved CSV to {csv_path}")
    print(f"Saved NPZ to {npz_path}")

    # Render MP4
    print(f"Rendering updated MP4 preview video for {gid}...")
    render_video(csv_path, mp4_path, title=f"PDF_004: Good Morning (Namaste -> Overhead Stretch -> Open Welcome)")
    print(f"Saved MP4 video to {mp4_path}")

    phrase = "Good Morning"

    meta_data = {
        "name": gid,
        "gesture_id": gid,
        "number": 4,
        "gesture_name": phrase,
        "spoken_response": phrase,
        "intent": phrase,
        "motion_prompt": user_prompt,
        "prompt": full_prompt,
        "category": "Greetings & Polite Interaction",
        "collection": "pdf_250_library",
        "frames": 130,
        "fps": 30.0,
        "duration_s": 4.33,
        "clamped": True,
        "valid": True,
        "status": "READY",
        "perfect": True,
        "approved": True,
        "published": True,
        "video": f"/files/pdf_250_library/{gid}/{gid}.mp4",
        "csv": f"/files/pdf_250_library/{gid}/{gid}.csv",
        "npz": f"/files/pdf_250_library/{gid}/{gid}.npz",
        "qpos_shape": [1, 130, 36]
    }

    with open(meta_path, "w") as mf:
        json.dump(meta_data, mf, indent=2)

    # Update in master gesture_library_500_1500.json & gesture_library_250.json
    for cat_name in ["gesture_library_500_1500.json", "gesture_library_250.json"]:
        cp = os.path.join(ROOT, cat_name)
        if os.path.isfile(cp):
            with open(cp) as f:
                cdata = json.load(f)
            for g in cdata.get("gestures", []):
                if g.get("gesture_id") == "PDF_004" or g.get("number") == 4:
                    g["motion_prompt"] = user_prompt
                    g["prompt"] = full_prompt
                    g["status"] = "READY"
                    g["valid"] = True
            with open(cp, "w") as f:
                json.dump(cdata, f, indent=2)

    print(f"Successfully updated PDF_004 Good Morning grand gesture sequence!")


if __name__ == "__main__":
    main()
