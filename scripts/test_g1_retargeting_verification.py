#!/usr/bin/env python3
"""Comprehensive System Verification Test for Unitree G1 Video-to-Gesture Pipeline."""

import os
import sys
import json
import subprocess
import numpy as np

_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_CUR_DIR) if os.path.basename(_CUR_DIR) == "scripts" else _CUR_DIR
SRC_DIR = os.path.join(ROOT, "src")
for _p in (ROOT, SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from g1_config import get_g1_config
from g1_robot_adapter import G1RobotAdapter
import video_gesture_pipeline as vgp

def main():
    print("====================================================================")
    print("UNITREE G1 VIDEO-TO-GESTURE RETARGETING PIPELINE VERIFICATION")
    print("====================================================================")

    # 1. Verify G1 Configuration
    config = get_g1_config()
    print(f"\n[1] Dynamic G1 Model Inspection:")
    print(f"    - Model Name: {config['model_name']}")
    print(f"    - Total DoF : {config['dof']}")
    print(f"    - Joints    : {config['total_joints']}")
    print(f"    - Controller: {config['controller_interface']}")
    assert config["dof"] == 36, "DoF count mismatch!"

    # 2. Test MediaPipe + IK Solver Retargeting on Video
    test_video = os.path.join(ROOT, "outputs", "web", "pdf_250_library", "PDF_001", "PDF_001.mp4")
    print(f"\n[2] Retargeting Input Video: {test_video}")
    res = vgp.process_video_to_gesture(test_video)

    print(f"    - Gesture ID   : {res['gesture_id']}")
    print(f"    - Name         : {res['gesture_name']}")
    print(f"    - Frames       : {res['frames']} ({res['duration_s']}s)")
    print(f"    - Quality Score: {res['quality_score']}/100")
    print(f"    - Similarity   : {res['similarity_report']['overall_similarity_percent']}%")

    assert res["success"] is True, "Pipeline failed to process video!"
    assert res["kinematic_valid"] is True, "Kinematic validation failed!"

    # 3. Structural Validation Pass via validate_g1_motion.py
    csv_file = os.path.join(ROOT, "outputs", "web", "pdf_250_library", res["gesture_id"], f"{res['gesture_id']}.csv")
    print(f"\n[3] Running Official Structural Kinematic Validation ({csv_file}):")
    val_script = os.path.join(SRC_DIR, "validate_g1_motion.py")
    if not os.path.isfile(val_script):
        val_script = os.path.join(ROOT, "validate_g1_motion.py")
    proc = subprocess.run([sys.executable, val_script, csv_file], capture_output=True, text=True)
    print(proc.stdout)
    assert proc.returncode == 0, f"Structural validation failed!\n{proc.stderr}"
    print("    PASSED ALL STRUCTURAL CHECKS!")

    # 4. Robot Adapter Execution Bridge Test
    adapter = G1RobotAdapter(mode="simulation_preview")
    readiness = adapter.check_robot_readiness()
    print(f"\n[4] Robot Adapter Readiness Check:")
    print(f"    - Ready : {readiness.get('ready', False)}")
    print(f"    - Status: {readiness.get('dof_status', 'OK')}")

    qpos = np.loadtxt(csv_file, delimiter=",")
    exec_res = adapter.execute_trajectory(qpos, user_approved=True)
    print(f"\n[5] Robot Adapter Execution Dispatch:")
    print(f"    - Executed Mode: {exec_res['mode']}")
    print(f"    - Frames       : {exec_res['frames']}")
    print(f"    - Message      : {exec_res['message']}")

    print("\n====================================================================")
    print("ALL VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("====================================================================")

if __name__ == "__main__":
    main()
