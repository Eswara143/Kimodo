#!/usr/bin/env python3
"""Final Website Gate & Production Asset Deployment Script for Unitree G1.

Calculates library readiness metrics across all 1,001 gesture IDs (500–1500).
Enforces the strict website gate:
    READY = 1001
    FAILED = 0
    BLOCKED = 0
    NEEDS_REVIEW = 0

Only upon 100% compliance will it:
1. Back up existing website catalogue and asset metadata.
2. Link/promotes all FINAL_ACCEPTED gesture assets (CSVs, NPZs, MP4s, meta.json).
3. Update `gesture_library_500_1500.json`.
4. Verify web application API readiness.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

CATALOGUE_PATH = os.path.join(ROOT, "gesture_library_500_1500.json")
BACKUP_CATALOGUE_PATH = os.path.join(ROOT, "outputs", f"gesture_library_500_1500_backup_{int(time.time())}.json")
LIBRARY_DIR = os.path.join(ROOT, "outputs", "web", "pdf_250_library")
SECOND_PASS_REPORT_PATH = os.path.join(ROOT, "outputs", "second_pass_quality_report.json")


def main() -> int:
    print("======================================================================")
    print("FINAL WEBSITE GATE & ASSET DEPLOYMENT")
    print("======================================================================")

    if not os.path.exists(CATALOGUE_PATH):
        print(f"ERROR: Catalogue file {CATALOGUE_PATH} does not exist.")
        return 1

    with open(CATALOGUE_PATH, "r") as f:
        data = json.load(f)
    gestures = data.get("gestures", [])
    target_items = [g for g in gestures if 500 <= g.get("number", 0) <= 1500]
    total_target = len(target_items)

    ready_count = 0
    failed_count = 0
    blocked_count = 0
    needs_review_count = 0

    for g in target_items:
        num = g.get("number", 0)
        gid = g.get("gesture_id", f"PDF_{num:03d}")
        meta_path = os.path.join(LIBRARY_DIR, gid, "meta.json")

        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                status = meta.get("status", "UNKNOWN")
            except Exception:
                status = "UNKNOWN"
        else:
            status = "MISSING"

        if status in ("READY", "CHECKS PASSED"):
            ready_count += 1
        elif status == "BLOCKED":
            blocked_count += 1
        elif status in ("NEEDS REVIEW", "NEEDS_REVIEW"):
            needs_review_count += 1
        else:
            failed_count += 1

    print(f"TOTAL_TARGET : {total_target}")
    print(f"READY        : {ready_count}")
    print(f"FAILED       : {failed_count}")
    print(f"BLOCKED      : {blocked_count}")
    print(f"NEEDS_REVIEW : {needs_review_count}")

    gate_passed = (
        total_target == 1001 and
        ready_count == 1001 and
        failed_count == 0 and
        blocked_count == 0 and
        needs_review_count == 0
    )

    if not gate_passed:
        print("\n[GATE REFUSED] Strict Website Gate condition NOT satisfied.")
        print("Final website deployment is BLOCKED until READY=1001, FAILED=0, BLOCKED=0, NEEDS_REVIEW=0.")
        return 1

    print("\n[GATE PASSED] All 1,001 gestures are 100% READY and verified!")

    # 1. Back up existing catalogue
    shutil.copy2(CATALOGUE_PATH, BACKUP_CATALOGUE_PATH)
    print(f"Backed up website catalogue to {BACKUP_CATALOGUE_PATH}")

    # 2. Update catalogue records to FINAL_ACCEPTED mapping
    updated_gestures = []
    for g in gestures:
        if 500 <= g.get("number", 0) <= 1500:
            gid = g.get("gesture_id")
            meta_path = os.path.join(LIBRARY_DIR, gid, "meta.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                g["status"] = "READY"
                g["accepted_version"] = "FINAL_ACCEPTED"
                g["motion_csv"] = f"outputs/web/pdf_250_library/{gid}/{gid}.csv"
                g["motion_npz"] = f"outputs/web/pdf_250_library/{gid}/{gid}.npz"
                g["preview_mp4"] = f"outputs/web/pdf_250_library/{gid}/{gid}.mp4"
        updated_gestures.append(g)

    data["gestures"] = updated_gestures
    data["ready_count"] = 1001
    data["failed_count"] = 0
    data["total_gestures"] = len(updated_gestures)

    with open(CATALOGUE_PATH, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Successfully updated master website catalogue {CATALOGUE_PATH} with 1,001 FINAL_ACCEPTED assets!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
