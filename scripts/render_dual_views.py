#!/usr/bin/env python3
"""Dual-View (Front & Side) Presentation Video Generator for Unitree G1.

Renders synchronized front (azimuth 165 deg) and side (azimuth 90 deg) preview videos
for a composed presentation sequence, verifying:
  - Higher forearm elevated talking pose
  - Clearer casual movements
  - Smooth phrase transitions
  - Planted feet / zero base drift
"""

from __future__ import annotations

import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import presentation_compose as pc

PRES_003_TEXT = ("Hello! Welcome to robotics. Let me explain about our robot. "
                 "Our robot can walk, run, jump, and perform a back flip. Thank you.")


def render_presentation_dual_views(out_dir: str, text: str = PRES_003_TEXT, name: str = "PRES_003"):
    os.makedirs(out_dir, exist_ok=True)
    print(f"Preparing presentation '{name}' in {out_dir}...")
    
    # Run standard prepare pass (renders front view preview.mp4)
    sched = pc.prepare(text, out_dir=out_dir, render=True, name=name)
    
    csv_path = os.path.join(out_dir, f"{name}.csv")
    raw_video_front = os.path.join(out_dir, "raw_front.mp4")
    raw_video_side = os.path.join(out_dir, "raw_side.mp4")
    wav_path = os.path.join(out_dir, "speech.wav")
    final_front_mp4 = os.path.join(out_dir, "preview_front.mp4")
    final_side_mp4 = os.path.join(out_dir, "preview_side.mp4")
    
    qpos = np.loadtxt(csv_path, delimiter=",").astype(np.float64)
    total_s = len(qpos) / pc.FPS
    
    print(f"Rendering Side View (azimuth = 90.0 deg)...")
    pc.render_paced(qpos, raw_video_side, azimuth=90.0)
    
    if os.path.exists(wav_path):
        pc.mux(raw_video_side, wav_path, final_side_mp4, total_s)
        # Also copy front preview to preview_front.mp4 for explicit side-by-side naming
        front_preview = os.path.join(out_dir, "preview.mp4")
        if os.path.exists(front_preview):
            import shutil
            shutil.copyfile(front_preview, final_front_mp4)
        print(f"Dual-view videos rendered:\n  - Front View: {final_front_mp4}\n  - Side View:  {final_side_mp4}")
    else:
        print(f"Audio file missing, rendered un-muxed side video at {raw_video_side}")
        
    return sched


if __name__ == "__main__":
    out_target = os.path.join(ROOT, "outputs", "presentations", "PRES_003_dual_view")
    render_presentation_dual_views(out_target)
