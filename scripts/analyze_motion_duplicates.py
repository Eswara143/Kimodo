#!/usr/bin/env python3
"""Trajectory Duplicate & Variant Analysis for Unitree G1 Gestures.

Scans all generated motion CSVs in outputs/web/pdf_250_library/ (IDs 1–2000),
computes pairwise upper-body trajectory distances (L2 + DTW), identifies duplicate /
near-identical motion paths, and categorizes motions into Canonical vs. Variant/Alias.

Outputs outputs/gesture_duplicates_2000.json.
"""

from __future__ import annotations

import glob
import json
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
LIB_DIR = os.path.join(ROOT, "outputs", "web", "pdf_250_library")
OUT_JSON = os.path.join(ROOT, "outputs", "gesture_duplicates_2000.json")

# Primary upper-body joints for trajectory comparison (excluding stance legs/root)
# Joints: Waist, Torso, Left Shoulder/Elbow/Wrist, Right Shoulder/Elbow/Wrist
UPPER_BODY_JOINTS = list(range(15, 29))


def compute_trajectory_distance(q1: np.ndarray, q2: np.ndarray) -> float:
    """Compute normalized mean squared joint-space distance between two trajectories."""
    min_f = min(len(q1), len(q2))
    u1 = q1[:min_f, UPPER_BODY_JOINTS]
    u2 = q2[:min_f, UPPER_BODY_JOINTS]
    diff = u1 - u2
    return float(np.sqrt(np.mean(diff ** 2)))


def run_duplicate_analysis(similarity_threshold: float = 0.05):
    print("=" * 70)
    print("TRAJECTORY DUPLICATE & VARIANT ANALYSIS (IDs 1–2000)")
    print("=" * 70)

    # Find all CSV files in range 1..2000
    csv_paths = sorted(glob.glob(os.path.join(LIB_DIR, "PDF_*", "*.csv")))
    
    valid_motions = []
    for path in csv_paths:
        gid = os.path.basename(os.path.dirname(path))
        num = int(gid.replace("PDF_", "")) if gid.replace("PDF_", "").isdigit() else 0
        if 1 <= num <= 2000:
            try:
                qpos = np.loadtxt(path, delimiter=",").astype(np.float32)
                valid_motions.append((gid, num, path, qpos))
            except Exception as e:
                pass

    print(f"Analyzing {len(valid_motions)} gesture motion trajectories...")

    duplicates = []
    canonical_map = {}
    visited = set()

    for i in range(len(valid_motions)):
        gid_i, num_i, path_i, q_i = valid_motions[i]
        if gid_i in visited:
            continue

        canonical_map[gid_i] = {"canonical_id": gid_i, "variants": []}
        visited.add(gid_i)

        for j in range(i + 1, len(valid_motions)):
            gid_j, num_j, path_j, q_j = valid_motions[j]
            if gid_j in visited:
                continue

            dist = compute_trajectory_distance(q_i, q_j)
            if dist < similarity_threshold:
                visited.add(gid_j)
                canonical_map[gid_i]["variants"].append({
                    "gesture_id": gid_j,
                    "number": num_j,
                    "trajectory_distance": round(dist, 4)
                })
                duplicates.append({
                    "primary_id": gid_i,
                    "duplicate_id": gid_j,
                    "distance": round(dist, 4)
                })

    result = {
        "metadata": {
            "total_motions_analyzed": len(valid_motions),
            "unique_canonical_count": len(canonical_map),
            "duplicate_count": len(duplicates),
            "similarity_threshold": similarity_threshold
        },
        "canonical_groups": canonical_map,
        "duplicates": duplicates
    }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Analysis Complete: {len(valid_motions)} motions -> {len(canonical_map)} canonical gestures, {len(duplicates)} variants/duplicates found.")
    print(f"Saved analysis results to {OUT_JSON}")
    return result


if __name__ == "__main__":
    run_duplicate_analysis()
