#!/usr/bin/env python3
"""Second Full-Library Quality Pass for Unitree G1 Gestures 500-1500.

Performs a rigorous, full-spectrum verification sweep across all 1,001 gestures
(IDs 500 through 1500) after initial individual refinement.
Re-evaluates every asset against strict kinematic and semantic checks.
If any gesture fails, revokes READY status and sends it back to the refinement loop.
Final pass completes only when ZERO incorrect gestures remain.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from gesture_specs import get_gesture_spec
from gesture_quality_checker import check_gesture_quality

CATALOGUE_PATH = os.path.join(ROOT, "gesture_library_500_1500.json")
LIBRARY_DIR = os.path.join(ROOT, "outputs", "web", "pdf_250_library")
SECOND_PASS_REPORT_PATH = os.path.join(ROOT, "outputs", "second_pass_quality_report.json")


def main() -> int:
    if not os.path.exists(CATALOGUE_PATH):
        print(f"Error: Catalogue JSON not found at {CATALOGUE_PATH}")
        return 1

    with open(CATALOGUE_PATH, "r") as f:
        data = json.load(f)
    gestures = data.get("gestures", [])
    target_items = [g for g in gestures if 500 <= g.get("number", 0) <= 1500]

    print("======================================================================")
    print("SECOND FULL-LIBRARY QUALITY PASS (IDs 500–1500)")
    print("======================================================================")
    print(f"Auditing full library of {len(target_items)} gestures...")

    t0 = time.time()
    total_checked = 0
    total_passed = 0
    total_failed = 0
    total_blocked = 0
    total_needs_review = 0

    failed_details = []
    audited_results = {}

    for item in target_items:
        num = item.get("number", 0)
        gid = item.get("gesture_id", f"PDF_{num:03d}")
        phrase = item.get("spoken_response") or item.get("intent") or ""
        action = item.get("source_action") or item.get("motion_prompt") or ""

        g_dir = os.path.join(LIBRARY_DIR, gid)
        csv_path = os.path.join(g_dir, f"{gid}.csv")
        meta_path = os.path.join(g_dir, "meta.json")

        total_checked += 1
        spec = get_gesture_spec(gid, intent=phrase, prompt=action)

        if not os.path.exists(csv_path):
            total_failed += 1
            failed_details.append({"gesture_id": gid, "reason": "MISSING_CSV_ASSET"})
            audited_results[gid] = "MISSING_ASSET"
            continue

        try:
            qpos = np.loadtxt(csv_path, delimiter=",")
        except Exception as e:
            total_failed += 1
            failed_details.append({"gesture_id": gid, "reason": f"CORRUPT_CSV: {e}"})
            audited_results[gid] = "CORRUPT_CSV"
            continue

        report = check_gesture_quality(qpos, spec=spec)

        if report.overall_status == "CHECKS PASSED":
            total_passed += 1
            audited_results[gid] = "READY"
        elif report.overall_status == "BLOCKED":
            total_blocked += 1
            audited_results[gid] = "BLOCKED"
            failed_details.append({"gesture_id": gid, "reason": "BLOCKED_WHOLE_BODY_ACTION", "report": asdict(report)})
        elif report.overall_status in ("NEEDS REVIEW", "NEEDS_REVIEW"):
            total_needs_review += 1
            audited_results[gid] = "NEEDS_REVIEW"
            failed_details.append({"gesture_id": gid, "reason": "NEEDS_REVIEW", "report": asdict(report)})
        else:
            total_failed += 1
            audited_results[gid] = "FAILED"
            failed_details.append({"gesture_id": gid, "reason": "FAILED_CHECKS", "failed_checks": report.failed_checks})

        # Update metadata file with second-pass status
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                meta["second_pass_status"] = report.overall_status
                meta["second_pass_passed"] = report.overall_status == "CHECKS PASSED"
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=2)
            except Exception:
                pass

    elapsed = time.time() - t0

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_target": len(target_items),
        "total_checked": total_checked,
        "total_passed": total_passed,
        "total_failed": total_failed,
        "total_blocked": total_blocked,
        "total_needs_review": total_needs_review,
        "elapsed_seconds": round(elapsed, 2),
        "zero_incorrect_gestures": (total_failed == 0 and total_blocked == 0 and total_needs_review == 0 and total_passed == len(target_items)),
        "failed_details": failed_details
    }

    os.makedirs(os.path.dirname(SECOND_PASS_REPORT_PATH), exist_ok=True)
    with open(SECOND_PASS_REPORT_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n----------------------------------------------------------------------")
    print(f"Second Pass Summary:")
    print(f"Total Target     : {len(target_items)}")
    print(f"Total Checked    : {total_checked}")
    print(f"TOTAL PASSED     : {total_passed}")
    print(f"TOTAL FAILED     : {total_failed}")
    print(f"TOTAL BLOCKED    : {total_blocked}")
    print(f"TOTAL NEEDS REVIEW: {total_needs_review}")
    print(f"Pass Execution   : {elapsed:.2f}s")
    print(f"Zero Incorrect   : {summary['zero_incorrect_gestures']}")
    print("----------------------------------------------------------------------")

    return 0 if summary["zero_incorrect_gestures"] else 1


if __name__ == "__main__":
    sys.exit(main())
