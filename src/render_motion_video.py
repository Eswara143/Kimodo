#!/usr/bin/env python3
"""Render a G1 qpos CSV to MP4 using MuJoCo's offscreen renderer.

Offscreen rendering needs a GL backend; EGL is used because it talks to the
NVIDIA driver directly and works without an X display, so this also runs over
SSH. PyAV does the encoding -- it is already a Kimodo dependency, so nothing new
is pulled in.

    python render_motion_video.py outputs/right_wave/right_wave.csv
    python render_motion_video.py --all
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

try:
    import av
    HAS_AV = True
except ImportError:
    HAS_AV = False
    import cv2

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from kimodo.assets import skeleton_asset_path  # noqa: E402


def render(csv_path: str, out_path: str, fps: float = 30.0,
           width: int = 640, height: int = 480, orbit: float = 50.0,
           title: str = "") -> str:
    model = mujoco.MjModel.from_xml_path(str(skeleton_asset_path("g1skel34", "xml", "g1.xml")))
    data = mujoco.MjData(model)
    qpos = np.loadtxt(csv_path, delimiter=",")
    if qpos.ndim == 1:
        qpos = qpos[None, :]

    renderer = mujoco.Renderer(model, height=height, width=width)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance = 3.0
    cam.elevation = -12.0
    cam.lookat[:] = [0.0, 0.0, 0.8]

    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    if HAS_AV:
        container = av.open(out_path, mode="w")
        stream = container.add_stream("libx264", rate=int(round(fps)))
        stream.width, stream.height = width, height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "20", "profile": "baseline", "preset": "veryfast"}

        n = len(qpos)
        for i, frame_qpos in enumerate(qpos):
            data.qpos[:] = frame_qpos
            mujoco.mj_forward(model, data)
            cam.azimuth = 135.0 + orbit * (i / max(n - 1, 1))
            cam.lookat[0] = float(frame_qpos[0])
            cam.lookat[1] = float(frame_qpos[1])
            renderer.update_scene(data, camera=cam, scene_option=opt)
            img = renderer.render()
            for packet in stream.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)
        container.close()
    else:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, float(fps), (width, height))

        n = len(qpos)
        for i, frame_qpos in enumerate(qpos):
            data.qpos[:] = frame_qpos
            mujoco.mj_forward(model, data)
            cam.azimuth = 135.0 + orbit * (i / max(n - 1, 1))
            cam.lookat[0] = float(frame_qpos[0])
            cam.lookat[1] = float(frame_qpos[1])
            renderer.update_scene(data, camera=cam, scene_option=opt)
            img = renderer.render()  # RGB uint8 (height, width, 3)
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            writer.write(bgr)

        writer.release()

    return out_path


render_video = render


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="?", help="qpos CSV to render")
    ap.add_argument("--all", action="store_true", help="Render every outputs/<gesture>/<gesture>.csv")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    targets: list[tuple[str, str]] = []
    if args.all:
        for csv in sorted(glob.glob("outputs/*/*.csv")):
            name = os.path.splitext(os.path.basename(csv))[0]
            if os.path.basename(os.path.dirname(csv)) != name:
                continue  # skip scratch dirs like pipeline_validation/
            targets.append((csv, f"outputs/videos/{name}.mp4"))
    elif args.csv:
        name = os.path.splitext(os.path.basename(args.csv))[0]
        targets.append((args.csv, args.out or f"outputs/videos/{name}.mp4"))
    else:
        ap.error("give a CSV path or --all")

    for csv, out in targets:
        print(f"rendering {csv} -> {out}", flush=True)
        render(csv, out, fps=args.fps)
        print(f"  {os.path.getsize(out) / 1e6:.2f} MB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
