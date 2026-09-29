#!/usr/bin/env python3
"""Offline safety validation for a Kimodo-generated G1 motion.

Checks a MuJoCo qpos CSV against the g1skel34 model before the motion is ever
considered for hardware. Everything here is read-only and simulation-side: no
robot transport is opened, and nothing is sent anywhere.

Usage:
    python validate_g1_motion.py <motion.csv> [--fps 30]

Exit code is 0 only when every check passes.
"""

from __future__ import annotations

import argparse
import sys

import mujoco
import numpy as np

from kimodo.assets import skeleton_asset_path

# Unitree G1 datasheet-style ceilings, deliberately conservative. These are
# review thresholds for flagging motions, not authorization to drive hardware.
MAX_JOINT_VEL_RAD_S = 10.0
MAX_JOINT_ACC_RAD_S2 = 200.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="MuJoCo qpos CSV produced by kimodo_gen / kimodo_convert")
    ap.add_argument("--fps", type=float, default=30.0, help="Playback frame rate (default: 30, Kimodo's native rate)")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(str(skeleton_asset_path("g1skel34", "xml", "g1.xml")))
    qpos = np.loadtxt(args.csv, delimiter=",")
    if qpos.ndim == 1:
        qpos = qpos[None, :]

    failures: list[str] = []
    warnings: list[str] = []

    print("=" * 68)
    print("G1 MOTION VALIDATION")
    print("=" * 68)
    print(f"CSV            : {args.csv}")
    print(f"MJCF           : {skeleton_asset_path('g1skel34', 'xml', 'g1.xml')}")
    print()

    # 1. Shape / dimension agreement with the model.
    n_frames, n_cols = qpos.shape
    print(f"[1] frames={n_frames}  columns={n_cols}  model.nq={model.nq}")
    if n_cols != model.nq:
        failures.append(f"qpos width {n_cols} != model.nq {model.nq}")
    else:
        print("    OK: qpos width matches model.nq")

    # 2. Duration and frame rate.
    duration = n_frames / args.fps
    print(f"[2] duration={duration:.3f}s @ {args.fps:g} fps")

    # 3. Numerical sanity.
    n_nan, n_inf = int(np.isnan(qpos).sum()), int(np.isinf(qpos).sum())
    print(f"[3] NaN={n_nan}  Inf={n_inf}")
    if n_nan or n_inf:
        failures.append(f"non-finite values present (NaN={n_nan}, Inf={n_inf})")
    else:
        print("    OK: all values finite")

    # 4. Root quaternion normalization (qpos[3:7] is the free joint's wxyz quat).
    if n_cols >= 7:
        norms = np.linalg.norm(qpos[:, 3:7], axis=1)
        worst = float(np.max(np.abs(norms - 1.0)))
        print(f"[4] root quaternion: max |norm-1| = {worst:.2e}")
        if worst > 1e-3:
            failures.append(f"root quaternion not unit-norm (max deviation {worst:.2e})")
        else:
            print("    OK: root orientation is a unit quaternion")

    # 5. Hinge joint limits, read from the model rather than hardcoded.
    hinge_qpos_cols, hinge_names, lo_lim, hi_lim = [], [], [], []
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE and model.jnt_limited[j]:
            hinge_qpos_cols.append(model.jnt_qposadr[j])
            hinge_names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j))
            lo_lim.append(model.jnt_range[j][0])
            hi_lim.append(model.jnt_range[j][1])

    print(f"[5] joint limits: checking {len(hinge_qpos_cols)} limited hinge joints")
    violations = 0
    for col, name, lo, hi in zip(hinge_qpos_cols, hinge_names, lo_lim, hi_lim):
        if col >= n_cols:
            continue
        vals = qpos[:, col]
        under, over = vals < lo, vals > hi
        if under.any() or over.any():
            violations += 1
            worst_lo, worst_hi = float(vals.min()), float(vals.max())
            print(f"    VIOLATION {name}: range [{worst_lo:+.3f},{worst_hi:+.3f}] vs limit [{lo:+.3f},{hi:+.3f}]")
    if violations:
        failures.append(f"{violations} joint(s) exceed MJCF limits")
    else:
        print("    OK: all joints within MJCF limits")

    # 6/7. Velocity and acceleration ceilings via finite differences.
    if n_frames >= 3 and hinge_qpos_cols:
        cols = [c for c in hinge_qpos_cols if c < n_cols]
        angles = qpos[:, cols]
        dt = 1.0 / args.fps
        vel = np.diff(angles, axis=0) / dt
        acc = np.diff(vel, axis=0) / dt
        max_v, max_a = float(np.abs(vel).max()), float(np.abs(acc).max())
        print(f"[6] max |joint velocity|     = {max_v:8.3f} rad/s   (threshold {MAX_JOINT_VEL_RAD_S})")
        print(f"[7] max |joint acceleration| = {max_a:8.3f} rad/s^2 (threshold {MAX_JOINT_ACC_RAD_S2})")
        if max_v > MAX_JOINT_VEL_RAD_S:
            warnings.append(f"peak joint velocity {max_v:.2f} rad/s exceeds {MAX_JOINT_VEL_RAD_S} rad/s")
        if max_a > MAX_JOINT_ACC_RAD_S2:
            warnings.append(f"peak joint acceleration {max_a:.2f} rad/s^2 exceeds {MAX_JOINT_ACC_RAD_S2} rad/s^2")

    # 8. Forward kinematics must succeed on every frame (structural validity).
    data = mujoco.MjData(model)
    fk_failed = 0
    for frame in qpos:
        data.qpos[:] = frame
        try:
            mujoco.mj_forward(model, data)
            if not np.all(np.isfinite(data.xpos)):
                fk_failed += 1
        except Exception:
            fk_failed += 1
    print(f"[8] forward kinematics: {n_frames - fk_failed}/{n_frames} frames resolved")
    if fk_failed:
        failures.append(f"forward kinematics produced non-finite body positions on {fk_failed} frame(s)")
    else:
        print("    OK: skeleton structurally valid on every frame")

    # 9. Root height sanity - catches a robot sunk through or launched off the floor.
    if n_cols >= 3:
        z = qpos[:, 2]
        print(f"[9] root height: min={z.min():.3f}m max={z.max():.3f}m mean={z.mean():.3f}m")
        if z.min() < 0.0:
            warnings.append(f"root dips below floor plane (min z = {z.min():.3f} m)")

    print()
    print("=" * 68)
    for w in warnings:
        print(f"WARNING: {w}")
    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        print("RESULT: FAILED - do not use this motion downstream")
        print("=" * 68)
        return 1
    print("RESULT: PASSED all structural checks")
    if warnings:
        print("        (warnings above require human review before any hardware use)")
    print("NOTE:   Simulation-side validation only. This is not hardware authorization.")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
