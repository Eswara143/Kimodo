#!/usr/bin/env python3
"""Dedicated Video -> Motion Analysis -> CSV Export -> G1 Gesture Converter CLI Tool.

Usage:
    python convert_video_to_csv_and_gesture.py <input_video_path>
"""

import os
import sys
import json
import numpy as np

_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_CUR_DIR) if os.path.basename(_CUR_DIR) == "src" else _CUR_DIR
for _p in (ROOT, _CUR_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import video_gesture_pipeline as vgp

def main():
    if len(sys.argv) < 2:
        print("Usage: python convert_video_to_csv_and_gesture.py <input_video_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    if not os.path.isfile(video_path):
        print(f"Error: Input video file '{video_path}' does not exist.")
        sys.exit(1)

    print("====================================================================")
    print("STEP 1: ANALYZING ENTIRE FRAME-BY-FRAME MOTION FROM INPUT VIDEO")
    print("====================================================================")
    print(f"Input Video Source File: {video_path}")

    def progress_callback(stage: int, stage_name: str, pct: float):
        print(f"  [{pct:5.1f}%] Stage {stage:2d}/12: {stage_name}")

    result = vgp.process_video_to_gesture(video_path, progress_callback=progress_callback)

    print("\n====================================================================")
    print("STEP 2: EXTRACTED MOTION CONVERTED TO STRUCTURED CSV FORMAT")
    print("====================================================================")
    print(f"CSV Trajectory File Path : {result['csv_path']}")
    print(f"Total Trajectory Frames  : {result['frames']}")
    print(f"Trajectory Frame Rate    : {result['fps']} FPS")
    print(f"Total Motion Duration    : {result['duration_s']} seconds")

    print("\n====================================================================")
    print("STEP 3: CONVERTED CSV TO ROBOT EXECUTABLE G1 GESTURE MOTION")
    print("====================================================================")
    print(f"Gesture ID               : {result['gesture_id']}")
    print(f"Gesture Name             : {result['gesture_name']}")
    print(f"G1 Robot Simulation MP4  : {result['video_path']}")
    print(f"NPZ Trajectory Artifact  : {result['npz_path']}")
    print(f"Kinematic Safety Valid   : {result['kinematic_valid']}")
    print(f"Overall Motion Similarity: {result['similarity_report']['overall_similarity_percent']}%")

    print("\nDetailed Motion Similarity Metrics:")
    for metric, val in result["similarity_report"].items():
        if isinstance(val, (int, float)):
            print(f"  - {metric:35s}: {val:.1f}%")

    print("\n====================================================================")
    print("PROCESS COMPLETED SUCCESSFULLY!")
    print("====================================================================")

if __name__ == "__main__":
    main()
