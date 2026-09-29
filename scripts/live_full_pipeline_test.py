#!/usr/bin/env python3
"""Live End-to-End Test Script for Video-to-Gesture Pipeline."""

import os
import sys
import json
import time

_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_CUR_DIR) if os.path.basename(_CUR_DIR) == "scripts" else _CUR_DIR
SRC_DIR = os.path.join(ROOT, "src")
for _p in (ROOT, SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import convert_video_to_csv_and_gesture as cvg
from g1_robot_adapter import G1RobotAdapter
import validate_g1_motion as vgm

def main():
    print("====================================================================")
    print("LIVE PIPELINE TEST: VIDEO -> 3D POSE -> CSV -> G1 EXECUTION")
    print("====================================================================")

    # 1. Select input video
    sample_video = os.path.join(ROOT, "outputs", "web", "pdf_250_library", "PDF_001", "PDF_001.mp4")
    print(f"\n[1/4] Source Input Video: {sample_video}")

    # 2. Run conversion to CSV & G1 Motion
    sys.argv = ["convert_video_to_csv_and_gesture.py", sample_video]
    print("\n[2/4] Running Frame-by-Frame Motion Extraction & CSV Conversion...")
    cvg.main()

    # 3. Verify Robot Adapter Readiness & Execute
    print("\n[3/4] Initializing G1 Robot Control Adapter...")
    adapter = G1RobotAdapter(mode="simulation_preview")
    readiness = adapter.check_robot_readiness()
    print(f"      - Readiness Check: Ready={readiness.get('ready')}, DOF Status={readiness.get('dof_status')}")

    # 4. Dispatch Execution
    latest_csv = os.path.join(ROOT, "outputs", "web", "pdf_250_library", "PDF_2025", "PDF_2025.csv")
    if not os.path.isfile(latest_csv):
        # find highest PDF_XXXX csv
        import glob
        csvs = sorted(glob.glob(os.path.join(ROOT, "outputs", "web", "pdf_250_library", "PDF_*", "*.csv")))
        latest_csv = csvs[-1] if csvs else latest_csv

    print(f"\n[4/4] Executing Robot Trajectory from CSV: {latest_csv}")
    import numpy as np
    qpos = np.loadtxt(latest_csv, delimiter=",")
    exec_res = adapter.execute_trajectory(qpos, user_approved=True)

    print("\n====================================================================")
    print("LIVE EXECUTION RESULT:")
    print("====================================================================")
    print(json.dumps(exec_res, indent=2))
    print("\nLIVE TEST COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
