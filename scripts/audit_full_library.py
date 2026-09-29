#!/usr/bin/env python3
"""Full Library Quality Audit & Targeted Correction Runner for Unitree G1.

Audits ALL 500 saved gesture assets in outputs/web/pdf_250_library/ PDF_001..PDF_500.
Evaluates kinematic, safety, semantic, and presentation checks against independent gesture specs.
Attempts up to 3 targeted corrections on failing gestures.
Generates outputs/full_library_audit.json and prints an honest, comprehensive summary report.
"""

from __future__ import annotations

import json
import glob
import os
import sys
import time
from dataclasses import asdict
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from gesture_specs import get_gesture_spec, KNOWN_SPECS, AMBIGUOUS_KEYWORDS
from gesture_quality_checker import check_gesture_quality
from gesture_corrector import attempt_gesture_correction

LIBRARY_DIR = os.path.join(ROOT, "outputs", "web", "pdf_250_library")
LIBRARY_JSON_PATH = os.path.join(ROOT, "gesture_library_250.json")
OUT_AUDIT_JSON = os.path.join(ROOT, "outputs", "full_library_audit.json")


def run_full_library_audit() -> dict:
    t0 = time.time()
    
    # Load library metadata mapping if available
    lib_meta = {}
    if os.path.exists(LIBRARY_JSON_PATH):
        try:
            with open(LIBRARY_JSON_PATH, "r") as f:
                data = json.load(f)
                for item in data.get("gestures", []):
                    lib_meta[item.get("gesture_id")] = item
        except Exception as e:
            print(f"Warning loading gesture_library_250.json: {e}")

    # Discover all gesture folders
    pdf_dirs = sorted(glob.glob(os.path.join(LIBRARY_DIR, "PDF_*")))
    total_discovered = len(pdf_dirs)
    print(f"Discovered {total_discovered} gesture directories in {LIBRARY_DIR}")

    audit_records = []
    
    counts = {
        "total_discovered": total_discovered,
        "total_tested": 0,
        "initial_passed": 0,
        "post_correction_passed": 0,
        "total_passed": 0,
        "total_failed": 0,
        "total_needs_review": 0,
        "total_blocked": 0,
        "total_not_tested": 0,
        "explicit_specs_count": 0,
        "inferred_specs_count": 0,
        "missing_independent_specs_count": 0,
    }

    failed_gestures_detail = []
    unresolved_gestures = []

    for idx, dpath in enumerate(pdf_dirs, 1):
        gid = os.path.basename(dpath)
        csv_file = os.path.join(dpath, f"{gid}.csv")
        
        if not os.path.exists(csv_file):
            counts["total_not_tested"] += 1
            audit_records.append({
                "gesture_id": gid,
                "status": "NOT_TESTED",
                "reason": f"Missing CSV file at {csv_file}"
            })
            continue

        counts["total_tested"] += 1
        
        # Load motion data
        try:
            qpos = np.loadtxt(csv_file, delimiter=",").astype(np.float64)
        except Exception as exc:
            counts["total_failed"] += 1
            audit_records.append({
                "gesture_id": gid,
                "status": "FAILED",
                "reason": f"Corrupt CSV data: {exc}"
            })
            continue

        # Get spec
        meta = lib_meta.get(gid, {})
        intent = meta.get("intent", gid)
        prompt = meta.get("motion_prompt", intent)
        spec = get_gesture_spec(gid, intent=intent, prompt=prompt)
        
        if gid in KNOWN_SPECS:
            counts["explicit_specs_count"] += 1
        else:
            counts["inferred_specs_count"] += 1

        # Check quality
        init_report = check_gesture_quality(qpos, spec=spec)
        init_status = init_report.overall_status

        rec = {
            "gesture_id": gid,
            "intent": intent,
            "prompt": prompt,
            "spec_type": "EXPLICIT" if gid in KNOWN_SPECS else "INFERRED",
            "initial_status": init_status,
            "final_status": init_status,
            "correction_attempts": 0,
            "correction_success": False,
            "failed_checks": [c["check_name"] for c in init_report.failed_checks],
            "details": asdict(init_report)
        }

        if init_status == "CHECKS PASSED":
            counts["initial_passed"] += 1
            counts["total_passed"] += 1
        elif init_status == "BLOCKED":
            counts["total_blocked"] += 1
        elif init_status == "NEEDS REVIEW":
            counts["total_needs_review"] += 1
            unresolved_gestures.append(gid)
        else:
            # Status is FAILED
            failed_gestures_detail.append({
                "gesture_id": gid,
                "intent": intent,
                "prompt": prompt,
                "failed_checks": [c["check_name"] for c in init_report.failed_checks],
                "failed_details": init_report.failed_checks
            })
            
            # Attempt correction (max 3 retries)
            corr_res = attempt_gesture_correction(gid, qpos, spec=spec, max_attempts=3)
            rec["correction_attempts"] = corr_res.attempts_made
            rec["correction_success"] = corr_res.success
            rec["final_status"] = corr_res.final_status
            rec["correction_history"] = corr_res.attempt_history

            if corr_res.success:
                counts["post_correction_passed"] += 1
                counts["total_passed"] += 1
            elif corr_res.final_status == "NEEDS REVIEW":
                counts["total_needs_review"] += 1
                unresolved_gestures.append(gid)
            else:
                counts["total_failed"] += 1
                unresolved_gestures.append(gid)

        audit_records.append(rec)

        if idx % 50 == 0 or idx == total_discovered:
            print(f"Audited {idx}/{total_discovered} gestures... (Passed: {counts['total_passed']}, Failed: {counts['total_failed']}, Needs Review: {counts['total_needs_review']}, Blocked: {counts['total_blocked']})")

    duration_s = time.time() - t0

    full_output = {
        "metadata": {
            "audit_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "duration_seconds": round(duration_s, 2),
            "checker_version": "v1.1.0",
            "model_version": "g1skel34_v1",
        },
        "summary": counts,
        "failed_gestures_detail": failed_gestures_detail,
        "unresolved_gestures": unresolved_gestures,
        "records": audit_records
    }

    os.makedirs(os.path.dirname(OUT_AUDIT_JSON), exist_ok=True)
    with open(OUT_AUDIT_JSON, "w") as f:
        json.dump(full_output, f, indent=2)

    print("\n" + "=" * 70)
    print("FULL LIBRARY GESTURE QUALITY AUDIT REPORT")
    print("=" * 70)
    print(f"Total Discovered Assets : {counts['total_discovered']}")
    print(f"Total Tested Assets     : {counts['total_tested']}")
    print(f"Initial Checks Passed  : {counts['initial_passed']}")
    print(f"Post-Correction Passed : {counts['post_correction_passed']}")
    print(f"Total Passed Assets     : {counts['total_passed']}")
    print(f"Total Failed Assets     : {counts['total_failed']}")
    print(f"Total Needs Review      : {counts['total_needs_review']}")
    print(f"Total Blocked (Static)  : {counts['total_blocked']}")
    print(f"Total Not Tested        : {counts['total_not_tested']}")
    print("-" * 70)
    print(f"Explicit Hand Specifications   : {counts['explicit_specs_count']}")
    print(f"Inferred Hand Specifications   : {counts['inferred_specs_count']}")
    print("-" * 70)

    return full_output


if __name__ == "__main__":
    run_full_library_audit()
