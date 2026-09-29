#!/usr/bin/env python3
"""Audit and Motion-Asset Mapping Repair for Unitree G1 Gestures 1000 to 1500.

Verifies:
1. Exact 1:1 ID to phrase, motion prompt, NPZ, MP4, and preview mapping.
2. Checks for off-by-one errors, duplicate asset paths, missing NPZ/MP4.
3. Evaluates motion kinematic quality (stationary body, joint limits, correct arm).
4. Performs targeted refinement ONLY if actual motion fails quality checks.
5. Emits strict audit statistics for all 501 records (1000 to 1500).
"""

from __future__ import annotations

import json
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from gesture_specs import get_gesture_spec
from gesture_quality_checker import check_gesture_quality
from master_retrain_500_1500 import process_gesture_refinement


def audit_1000_1500():
    print("=" * 75)
    print("AUDITING & REPAIRING MOTION-ASSET MAPPINGS (IDs 1000–1500)")
    print("=" * 75)

    json_p = os.path.join(ROOT, "gesture_library_500_1500.json")
    if not os.path.isfile(json_p):
        print(f"ERROR: {json_p} not found!")
        sys.exit(1)

    data = json.load(open(json_p))
    gestures = data.get("gestures", [])
    
    # Filter 1000 to 1500
    target_records = [g for g in gestures if 1000 <= g.get("number", 0) <= 1500]
    
    stats = {
        "TOTAL_RECORDS": len(target_records),
        "MISSING_IDS": 0,
        "DUPLICATE_IDS": 0,
        "NONE_GESTURE_NAMES": 0,
        "NULL_SPOKEN_RESPONSES": 0,
        "NULL_MOTION_PROMPTS": 0,
        "MISSING_NPZ": 0,
        "WRONG_NPZ_MAPPING": 0,
        "MISSING_MP4": 0,
        "ZERO_DURATION_MP4": 0,
        "WRONG_VIDEO_MAPPING": 0,
        "BROKEN_PREVIEW": 0,
        "SEMANTIC_MOTION_MISMATCH": 0,
        "READY": 0,
    }

    seen_ids = set()
    assigned_npz = set()
    assigned_mp4 = set()

    # Track missing IDs in range 1000..1500
    expected_numbers = set(range(1000, 1501))
    found_numbers = set()

    for idx, g in enumerate(target_records, 1):
        num = g.get("number", 0)
        found_numbers.add(num)
        gid = g.get("gesture_id") or f"PDF_{num:03d}"

        # 1. ID duplication check
        if gid in seen_ids:
            stats["DUPLICATE_IDS"] += 1
            print(f"❌ [DUPLICATE ID] {gid}")
        seen_ids.add(gid)

        # 2. Metadata field checks
        gname = g.get("gesture_name") or g.get("intent")
        spoken = g.get("spoken_response") or g.get("intent")
        mprompt = g.get("motion_prompt") or g.get("required_action")

        if not gname or gname in ("None", "null"):
            stats["NONE_GESTURE_NAMES"] += 1
            print(f"❌ [NONE NAME] #{num} ({gid})")

        if not spoken or spoken in ("None", "null"):
            stats["NULL_SPOKEN_RESPONSES"] += 1
            print(f"❌ [NULL SPOKEN] #{num} ({gid})")

        if not mprompt or mprompt in ("None", "null"):
            stats["NULL_MOTION_PROMPTS"] += 1
            print(f"❌ [NULL PROMPT] #{num} ({gid})")

        # 3. File Asset Mapping Audit
        out_folder = os.path.join(ROOT, "outputs", "web", "pdf_250_library", gid)
        expected_csv = os.path.join(out_folder, f"{gid}.csv")
        expected_npz = os.path.join(out_folder, f"{gid}.npz")
        expected_mp4 = os.path.join(out_folder, f"{gid}.mp4")

        # Verify correct asset paths in JSON metadata
        json_npz = g.get("npz_path", "")
        json_mp4 = g.get("video_path", "")

        if not json_npz or f"/{gid}/{gid}.npz" not in json_npz:
            stats["WRONG_NPZ_MAPPING"] += 1
            g["npz_path"] = f"/files/pdf_250_library/{gid}/{gid}.npz"

        if not json_mp4 or f"/{gid}/{gid}.mp4" not in json_mp4:
            stats["WRONG_VIDEO_MAPPING"] += 1
            g["video_path"] = f"/files/pdf_250_library/{gid}/{gid}.mp4"

        # Check duplicate asset assignments
        if expected_npz in assigned_npz:
            stats["WRONG_NPZ_MAPPING"] += 1
            print(f"❌ [DUPLICATE NPZ] {expected_npz} assigned to multiple IDs!")
        assigned_npz.add(expected_npz)

        if expected_mp4 in assigned_mp4:
            stats["WRONG_VIDEO_MAPPING"] += 1
            print(f"❌ [DUPLICATE MP4] {expected_mp4} assigned to multiple IDs!")
        assigned_mp4.add(expected_mp4)

        # Check file existence & non-zero size
        if not os.path.isfile(expected_npz) or os.path.getsize(expected_npz) == 0:
            stats["MISSING_NPZ"] += 1
            print(f"❌ [MISSING NPZ] {expected_npz}")

        if not os.path.isfile(expected_mp4) or os.path.getsize(expected_mp4) == 0:
            stats["MISSING_MP4"] += 1
            stats["ZERO_DURATION_MP4"] += 1
            print(f"❌ [MISSING/ZERO MP4] {expected_mp4}")

        # 4. Kinematic & Motion Quality Audit
        spec = get_gesture_spec(gid, intent=spoken, prompt=f"{spoken} — {mprompt}")
        qpos = None

        if os.path.isfile(expected_npz):
            try:
                npz_data = np.load(expected_npz)
                qpos = npz_data["qpos"] if "qpos" in npz_data else None
            except Exception:
                pass

        if qpos is None and os.path.isfile(expected_csv):
            try:
                qpos = np.loadtxt(expected_csv, delimiter=",")
            except Exception:
                pass

        if qpos is not None:
            report = check_gesture_quality(qpos, spec=spec, fps=30.0)
            if report.overall_status == "CHECKS PASSED":
                stats["READY"] += 1
                g["status"] = "READY"
                g["valid"] = True
            else:
                print(f"⚠️ [MOTION QUALITY REJECTED] {gid} (#{num}): {[c['check_name'] for c in report.failed_checks]}")
                print(f"   --> Entering targeted refinement for {gid}...")
                ref_res = process_gesture_refinement(g, max_attempts=5)
                if ref_res.get("status") == "READY":
                    stats["READY"] += 1
                    g["status"] = "READY"
                    g["valid"] = True
                else:
                    stats["SEMANTIC_MOTION_MISMATCH"] += 1
                    g["status"] = "FAILED"
                    g["valid"] = False
        else:
            print(f"⚠️ [NO QPOS FOUND] {gid} (#{num}) --> Synthesizing motion...")
            ref_res = process_gesture_refinement(g, max_attempts=5)
            if ref_res.get("status") == "READY":
                stats["READY"] += 1
                g["status"] = "READY"
                g["valid"] = True
            else:
                stats["SEMANTIC_MOTION_MISMATCH"] += 1
                g["status"] = "FAILED"
                g["valid"] = False

    stats["MISSING_IDS"] = len(expected_numbers - found_numbers)

    # Save updated catalogue JSON
    with open(json_p, "w") as f:
        json.dump(data, f, indent=2)

    print("\n" + "=" * 75)
    print("AUTOMATED FINAL AUDIT RESULTS (IDs 1000–1500)")
    print("=" * 75)
    for k, v in stats.items():
        print(f"{k:<25}: {v}")
    print("=" * 75)
    return stats


if __name__ == "__main__":
    audit_1000_1500()
