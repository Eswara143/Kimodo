#!/usr/bin/env python3
"""Gesture Quality & Kinematic Checker for Unitree G1.

Inspects generated or saved gestures against their independent specifications using
MuJoCo forward kinematics (g1skel34), joint mapping, time-based physical velocity/accel
thresholds (rad/s, rad/s^2), and collision physics.

Evaluates presentation timelines separately (talking pose transitions, phrase alignment,
restricted action blocking).

Reports failed checks with measured values, thresholds, and affected frame/timestamp ranges.
Explicitly labels dynamic simulation checks as NOT TESTED (FK Kinematic Evaluation Only).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass, field, asdict

import numpy as np

_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_CUR_DIR) if os.path.basename(_CUR_DIR) == "src" else _CUR_DIR
for _p in (ROOT, _CUR_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import presentation_safety as ps
from gesture_specs import GestureSpec, get_gesture_spec

CHECKER_VERSION = "v1.1.0"
MODEL_VERSION = "g1skel34_v1"

UNAVAILABLE_CHECKS = [
    {"name": "Dynamic Motor Torque Saturation", "reason": "Requires real-time hardware motor current telemetry", "status": "NOT TESTED (FK Kinematic Evaluation Only)"},
    {"name": "Ground Friction Cone & Contact Wrench", "reason": "Requires high-frequency force-torque foot sensors", "status": "NOT TESTED (FK Kinematic Evaluation Only)"},
    {"name": "Hardware Actuator Thermal Ceiling", "reason": "Requires hardware motor thermistor logs", "status": "NOT TESTED (FK Kinematic Evaluation Only)"}
]


@dataclass
class QualityCheckItem:
    check_name: str
    category: str                  # 'kinematic', 'safety', 'semantic', 'presentation'
    passed: bool
    status: str                    # 'PASSED', 'FAILED', 'NEEDS_REVIEW', 'BLOCKED', 'NOT_TESTED'
    measured_value: str
    threshold: str
    frame_range: list[int]         # [start_frame, end_frame]
    timestamp_range_s: list[float] # [start_s, end_s]
    detail: str


@dataclass
class QualityReport:
    gesture_id: str
    intent: str
    prompt: str
    overall_status: str            # 'CHECKS PASSED', 'FAILED', 'NEEDS REVIEW', 'BLOCKED', 'NOT TESTED'
    spec: dict
    checks: list[dict]
    failed_checks: list[dict]
    unavailable_checks: list[dict]
    asset_hash: str
    checker_version: str
    model_version: str
    observed_metrics: dict
    correction_history: list[dict] = field(default_factory=list)


def _compute_asset_hash(qpos: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(qpos).tobytes()).hexdigest()[:16]


def _count_wave_cycles(signal: np.ndarray, min_prominence: float = 0.015) -> int:
    """Count peak-to-trough cycles in a 1D hand displacement waveform."""
    if len(signal) < 10:
        return 0
    s = signal - np.mean(signal)
    peaks, troughs = [], []
    for i in range(1, len(s) - 1):
        if s[i] > s[i - 1] and s[i] > s[i + 1] and s[i] > min_prominence:
            peaks.append(i)
        elif s[i] < s[i - 1] and s[i] < s[i + 1] and s[i] < -min_prominence:
            troughs.append(i)
    return max(len(peaks), len(troughs))


def check_gesture_quality(qpos: np.ndarray, spec: GestureSpec | None = None,
                          gesture_id: str = "G_UNKNOWN", intent: str = "", prompt: str = "",
                          fps: float = 30.0) -> QualityReport:
    """Comprehensive kinematic, safety, and semantic quality inspection for one gesture qpos."""
    qpos = np.asarray(qpos, dtype=np.float64)
    if qpos.ndim == 1:
        qpos = qpos[None, :]

    if spec is None:
        spec = get_gesture_spec(gesture_id, intent=intent, prompt=prompt)

    asset_hash = _compute_asset_hash(qpos)
    n_frames = len(qpos)
    duration_s = n_frames / fps

    checks: list[QualityCheckItem] = []
    observed_metrics: dict = {}

    # 1. Data Integrity & Finite Values
    finite_ok = bool(np.all(np.isfinite(qpos)))
    checks.append(QualityCheckItem(
        check_name="Finite Motion Data",
        category="kinematic",
        passed=finite_ok,
        status="PASSED" if finite_ok else "FAILED",
        measured_value="Finite" if finite_ok else "Non-finite (NaN/Inf)",
        threshold="All values finite",
        frame_range=[0, n_frames - 1],
        timestamp_range_s=[0.0, round(duration_s, 3)],
        detail="Valid numeric floating point array." if finite_ok else "Motion data contains NaN or Inf values."
    ))

    if not finite_ok:
        return QualityReport(
            gesture_id=spec.gesture_id, intent=spec.intent, prompt=spec.prompt,
            overall_status="FAILED", spec=asdict(spec),
            checks=[asdict(c) for c in checks],
            failed_checks=[asdict(c) for c in checks if not c.passed],
            unavailable_checks=UNAVAILABLE_CHECKS, asset_hash=asset_hash,
            checker_version=CHECKER_VERSION, model_version=MODEL_VERSION,
            observed_metrics={}
        )

    # 2. Time-Based Joint Continuity & Rates (rad/s, rad/s^2)
    hinges = qpos[:, 7:36]
    dt = 1.0 / fps
    vel = np.diff(hinges, axis=0) / dt if n_frames >= 2 else np.zeros((1, hinges.shape[1]))
    acc = np.diff(vel, axis=0) / dt if n_frames >= 3 else np.zeros((1, hinges.shape[1]))

    max_vel_rad_s = float(np.abs(vel).max()) if vel.size else 0.0
    max_acc_rad_s2 = float(np.abs(acc).max()) if acc.size else 0.0

    step_diffs_rad = np.abs(np.diff(hinges, axis=0)) if n_frames >= 2 else np.zeros((0, 29))
    max_step_rad = float(step_diffs_rad.max()) if step_diffs_rad.size else 0.0
    max_step_vel_rad_s = max_step_rad * fps
    jump_frame = int(np.unravel_index(step_diffs_rad.argmax(), step_diffs_rad.shape)[0]) + 1 if step_diffs_rad.size else 0

    # Physical continuity: step jump rate <= 15.0 rad/s (~0.5 rad/frame at 30fps)
    jump_ok = max_step_vel_rad_s <= 15.0
    checks.append(QualityCheckItem(
        check_name="Time-Based Joint Continuity",
        category="kinematic",
        passed=jump_ok,
        status="PASSED" if jump_ok else "FAILED",
        measured_value=f"{max_step_vel_rad_s:.2f} rad/s ({max_step_rad:.3f} rad/frame)",
        threshold="<= 15.00 rad/s (0.500 rad/frame at 30 fps)",
        frame_range=[max(0, jump_frame - 1), min(n_frames - 1, jump_frame + 1)],
        timestamp_range_s=[round(max(0, jump_frame - 1) / fps, 3), round(min(n_frames - 1, jump_frame + 1) / fps, 3)],
        detail="Joint trajectory continuous in time." if jump_ok else f"Sudden joint step jump rate of {max_step_vel_rad_s:.2f} rad/s at t={jump_frame/fps:.2f}s (frame {jump_frame})."
    ))

    # Joint Velocity & Acceleration Limits
    vel_ok = max_vel_rad_s <= ps.THRESHOLDS["max_joint_vel_rad_s"]
    checks.append(QualityCheckItem(
        check_name="Joint Velocity Ceiling",
        category="safety",
        passed=vel_ok,
        status="PASSED" if vel_ok else "FAILED",
        measured_value=f"{max_vel_rad_s:.2f} rad/s",
        threshold=f"<= {ps.THRESHOLDS['max_joint_vel_rad_s']:.1f} rad/s",
        frame_range=[0, n_frames - 1],
        timestamp_range_s=[0.0, round(duration_s, 3)],
        detail="Joint velocities within physical limits." if vel_ok else f"Max joint velocity {max_vel_rad_s:.2f} rad/s exceeds hard limit {ps.THRESHOLDS['max_joint_vel_rad_s']:.1f} rad/s."
    ))

    acc_ok = max_acc_rad_s2 <= ps.THRESHOLDS["max_joint_acc_rad_s2"]
    checks.append(QualityCheckItem(
        check_name="Joint Acceleration Ceiling",
        category="safety",
        passed=acc_ok,
        status="PASSED" if acc_ok else "FAILED",
        measured_value=f"{max_acc_rad_s2:.2f} rad/s²",
        threshold=f"<= {ps.THRESHOLDS['max_joint_acc_rad_s2']:.1f} rad/s²",
        frame_range=[0, n_frames - 1],
        timestamp_range_s=[0.0, round(duration_s, 3)],
        detail="Joint accelerations within physical limits." if acc_ok else f"Max joint acceleration {max_acc_rad_s2:.2f} rad/s² exceeds hard limit {ps.THRESHOLDS['max_joint_acc_rad_s2']:.1f} rad/s²."
    ))

    # 3. Forward Kinematics & Range of Motion Analysis
    ctx = ps.load_model()
    model, data, mujoco = ctx["model"], ctx["data"], ctx["mujoco"]
    l_hand_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link")
    r_hand_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")

    r_hand_pos = np.empty((n_frames, 3))
    l_hand_pos = np.empty((n_frames, 3))

    for i in range(n_frames):
        data.qpos[:] = qpos[i]
        mujoco.mj_forward(model, data)
        r_hand_pos[i] = data.xpos[r_hand_body]
        l_hand_pos[i] = data.xpos[l_hand_body]

    l_arm_rom = float(np.ptp(qpos[:, 22:29], axis=0).max())
    r_arm_rom = float(np.ptp(qpos[:, 29:36], axis=0).max())
    observed_metrics["left_arm_rom_rad"] = round(l_arm_rom, 3)
    observed_metrics["right_arm_rom_rad"] = round(r_arm_rom, 3)

    if spec.arm == "right":
        arm_ok = r_arm_rom >= spec.min_arm_rom_rad and (r_arm_rom > l_arm_rom - 0.10)
        arm_msg = f"Right arm ROM {r_arm_rom:.2f} rad, Left arm ROM {l_arm_rom:.2f} rad"
    elif spec.arm == "left":
        arm_ok = l_arm_rom >= spec.min_arm_rom_rad and (l_arm_rom > r_arm_rom - 0.10)
        arm_msg = f"Left arm ROM {l_arm_rom:.2f} rad, Right arm ROM {r_arm_rom:.2f} rad"
    elif spec.arm == "both":
        arm_ok = r_arm_rom >= spec.min_arm_rom_rad and l_arm_rom >= spec.min_arm_rom_rad
        arm_msg = f"Both arms ROM: Right {r_arm_rom:.2f} rad, Left {l_arm_rom:.2f} rad"
    else:
        arm_ok = True
        arm_msg = f"Either arm active: Right {r_arm_rom:.2f}, Left {l_arm_rom:.2f}"

    checks.append(QualityCheckItem(
        check_name="Correct Arm Execution",
        category="semantic",
        passed=arm_ok,
        status="PASSED" if arm_ok else "FAILED",
        measured_value=arm_msg,
        threshold=f"Expected {spec.arm.upper()} arm active (min ROM >= {spec.min_arm_rom_rad:.2f} rad)",
        frame_range=[0, n_frames - 1],
        timestamp_range_s=[0.0, round(duration_s, 3)],
        detail=f"Correct arm ({spec.arm}) executed primary movement." if arm_ok else f"Incorrect arm used or movement too weak for {spec.arm} arm."
    ))

    # 4. Wave Repetition Count
    if spec.expected_wave_cycles > 0:
        signal = r_hand_pos[:, 1] if spec.arm in ("right", "both") else l_hand_pos[:, 1]
        detected_cycles = _count_wave_cycles(signal)
        observed_metrics["detected_wave_cycles"] = detected_cycles
        cycles_ok = detected_cycles >= spec.expected_wave_cycles
        checks.append(QualityCheckItem(
            check_name="Wave Repetition Count",
            category="semantic",
            passed=cycles_ok,
            status="PASSED" if cycles_ok else "FAILED",
            measured_value=f"{detected_cycles} wave cycles detected",
            threshold=f">= {spec.expected_wave_cycles} wave cycles",
            frame_range=[0, n_frames - 1],
            timestamp_range_s=[0.0, round(duration_s, 3)],
            detail=f"Detected {detected_cycles} distinct wave cycles." if cycles_ok else f"Expected {spec.expected_wave_cycles} wave cycles, but only detected {detected_cycles}."
        ))

    # 5. Hold Duration Check
    prim_arm_slice = slice(29, 36) if spec.arm == "right" else slice(22, 29)
    arm_vel = np.abs(np.diff(qpos[:, prim_arm_slice], axis=0)) * fps if n_frames >= 2 else np.zeros((1, 7))
    low_vel_frames = np.sum(arm_vel.max(axis=1) < 0.25)
    hold_duration_s = float(low_vel_frames / fps)
    observed_metrics["hold_duration_s"] = round(hold_duration_s, 2)
    hold_ok = hold_duration_s >= spec.min_hold_seconds or spec.min_hold_seconds <= 0.1
    checks.append(QualityCheckItem(
        check_name="Gesture Hold Duration",
        category="kinematic",
        passed=hold_ok,
        status="PASSED" if hold_ok else "FAILED",
        measured_value=f"{hold_duration_s:.2f} s hold",
        threshold=f">= {spec.min_hold_seconds:.2f} s hold",
        frame_range=[0, n_frames - 1],
        timestamp_range_s=[0.0, round(duration_s, 3)],
        detail="Gesture held pose sufficiently at peak." if hold_ok else f"Gesture did not hold posture long enough at peak ({hold_duration_s:.2f}s < {spec.min_hold_seconds:.2f}s)."
    ))

    # 6. Safety & Physical Balance Checks
    meas = ps.measure(qpos, fps=fps)
    observed_metrics.update({
        "foot_lift_m": round(meas["foot_lift_m"], 4),
        "foot_slide_m": round(meas["foot_slide_m"], 4),
        "root_drift_m": round(meas["root_drift_m"], 4),
        "self_penetration_excess_m": round(meas["self_penetration_excess_m"], 4),
        "max_joint_vel_rad_s": round(meas["max_joint_vel_rad_s"], 2),
        "max_joint_acc_rad_s2": round(meas["max_joint_acc_rad_s2"], 2),
    })

    feet_ok = meas["foot_lift_m"] <= 0.015 and meas["foot_slide_m"] <= 0.030 and meas["root_drift_m"] <= 0.060
    checks.append(QualityCheckItem(
        check_name="Stationary Feet & Base Stability",
        category="safety",
        passed=feet_ok,
        status="PASSED" if feet_ok else "FAILED",
        measured_value=f"lift {meas['foot_lift_m']*100:.1f}cm, slide {meas['foot_slide_m']*100:.1f}cm, drift {meas['root_drift_m']*100:.1f}cm",
        threshold="lift <= 1.5cm, slide <= 3.0cm, drift <= 6.0cm",
        frame_range=[0, n_frames - 1],
        timestamp_range_s=[0.0, round(duration_s, 3)],
        detail="Feet remained planted on the ground." if feet_ok else "Robot feet lifted, slid, or pelvis drifted during gesture."
    ))

    # 6b. Hand-Only Torso & Lower-Body Joint Stillness Check
    if spec.presentation_allowed:
        lower_body_stds = np.std(qpos[:, 7:22], axis=0) if n_frames >= 2 else np.zeros(15)
        max_body_motion_rad = float(lower_body_stds.max())
        body_still_ok = max_body_motion_rad <= 0.080
        observed_metrics["max_body_motion_std_rad"] = round(max_body_motion_rad, 4)
        checks.append(QualityCheckItem(
            check_name="Hand-Only Torso & Lower-Body Stillness",
            category="safety",
            passed=body_still_ok,
            status="PASSED" if body_still_ok else "NEEDS_REVIEW",
            measured_value=f"max lower body std {max_body_motion_rad:.3f} rad",
            threshold="<= 0.080 rad std across torso, waist, and leg joints",
            frame_range=[0, n_frames - 1],
            timestamp_range_s=[0.0, round(duration_s, 3)],
            detail="Torso, waist, and lower body remained stationary." if body_still_ok else f"Unintended body movement detected (std {max_body_motion_rad:.3f} rad > 0.080 rad)."
        ))

    col_ok = meas["self_penetration_excess_m"] <= 0.0
    checks.append(QualityCheckItem(
        check_name="Self-Collision & Clearance",
        category="safety",
        passed=col_ok,
        status="PASSED" if col_ok else "FAILED",
        measured_value=f"excess {meas['self_penetration_excess_m']*100:.1f}cm ({meas.get('self_penetration_pair', 'none')})",
        threshold="excess <= 0.0cm",
        frame_range=[meas.get("self_penetration_frame", 0), meas.get("self_penetration_frame", 0)],
        timestamp_range_s=[round(meas.get("self_penetration_frame", 0) / fps, 3), round(meas.get("self_penetration_frame", 0) / fps, 3)],
        detail="Zero self-collision detected." if col_ok else f"Self-collision detected between {meas.get('self_penetration_pair')}."
    ))

    limit_violations = ps.joint_limit_violations(qpos)
    limits_ok = len(limit_violations) == 0
    checks.append(QualityCheckItem(
        check_name="Hardware Joint Limits",
        category="safety",
        passed=limits_ok,
        status="PASSED" if limits_ok else "FAILED",
        measured_value=f"{len(limit_violations)} violation(s)",
        threshold="0 violations",
        frame_range=[0, n_frames - 1],
        timestamp_range_s=[0.0, round(duration_s, 3)],
        detail="All joints remained within hardware range." if limits_ok else "; ".join(limit_violations[:2])
    ))

    pres_ok = spec.presentation_allowed
    checks.append(QualityCheckItem(
        check_name="Presentation Mode Allowed",
        category="semantic",
        passed=pres_ok,
        status="PASSED" if pres_ok else "BLOCKED",
        measured_value="Allowed" if pres_ok else "Blocked (Locomotion/Whole-body action)",
        threshold="Allowed stationary gesture",
        frame_range=[0, n_frames - 1],
        timestamp_range_s=[0.0, round(duration_s, 3)],
        detail="Gesture is allowed in presentation speech mode." if pres_ok else "Gesture intent represents a whole-body action (walk/run/jump/flip/sit)."
    ))

    failed_items = [c for c in checks if not c.passed]
    if spec.spec_status == "NEEDS_REVIEW":
        overall_status = "NEEDS REVIEW"
    elif not pres_ok:
        overall_status = "BLOCKED"
    elif failed_items:
        overall_status = "FAILED"
    else:
        overall_status = "CHECKS PASSED"

    return QualityReport(
        gesture_id=spec.gesture_id,
        intent=spec.intent,
        prompt=spec.prompt,
        overall_status=overall_status,
        spec=asdict(spec),
        checks=[asdict(c) for c in checks],
        failed_checks=[asdict(c) for c in failed_items],
        unavailable_checks=UNAVAILABLE_CHECKS,
        asset_hash=asset_hash,
        checker_version=CHECKER_VERSION,
        model_version=MODEL_VERSION,
        observed_metrics=observed_metrics
    )


def check_presentation_timeline_quality(schedule: dict, qpos_timeline: np.ndarray | None = None) -> list[QualityCheckItem]:
    """Inspect a composed presentation timeline for speech-gesture timing alignment and transitions."""
    items = []
    gestures = schedule.get("gestures", [])
    words = schedule.get("words", [])

    # 1. Main Action Speech Alignment Check (|lag| <= 250 ms)
    lag_failures = []
    for g in gestures:
        lag_s = g.get("main_lag_s", 0.0)
        if abs(lag_s) > 0.250:
            lag_failures.append(f"'{g.get('phrase')}' ({g.get('intent')}): lag {lag_s*1000:+.0f} ms > 250 ms")

    lag_ok = len(lag_failures) == 0
    items.append(QualityCheckItem(
        check_name="Speech/Gesture Main Action Synchronization",
        category="presentation",
        passed=lag_ok,
        status="PASSED" if lag_ok else "FAILED",
        measured_value="Aligned" if lag_ok else f"{len(lag_failures)} timing offset failure(s)",
        threshold="Speech-gesture main action lag <= 250 ms",
        frame_range=[0, 0],
        timestamp_range_s=[0.0, round(schedule.get("duration_s", 0.0), 3)],
        detail="All gestures aligned with spoken onset within 250 ms." if lag_ok else "; ".join(lag_failures)
    ))

    # 2. Inter-Phrase Attention Dip Verification
    segments = schedule.get("segments", [])
    gsegs = [seg for seg in segments if seg.get("kind") == "gesture"]
    dips_to_attention = 0
    if len(gsegs) >= 2 and qpos_timeline is not None:
        lo_f, hi_f = gsegs[0]["end_f"], gsegs[-1]["start_f"]
        between = [s for s in segments if s.get("kind") in ("talking", "transition", "exit", "entry")
                   and s.get("start_f") >= lo_f and s.get("end_f") <= hi_f]
        att_pose = schedule.get("attention_pose", np.zeros(36))
        for seg in between:
            block = qpos_timeline[seg["start_f"]: seg["end_f"]]
            if len(block) and np.min(np.linalg.norm(block[:, 22:36] - att_pose[22:36], axis=1)) < 0.15:
                dips_to_attention += 1

    dips_ok = dips_to_attention == 0
    items.append(QualityCheckItem(
        check_name="Talking Pose Transition Continuity (No Intermediate Attention Dips)",
        category="presentation",
        passed=dips_ok,
        status="PASSED" if dips_ok else "FAILED",
        measured_value=f"{dips_to_attention} inter-phrase attention dip(s)",
        threshold="0 dips to attention between intermediate gestures",
        frame_range=[0, 0],
        timestamp_range_s=[0.0, round(schedule.get("duration_s", 0.0), 3)],
        detail="Transitions maintained relaxed talking pose between spoken phrases." if dips_ok else f"{dips_to_attention} inter-phrase transition(s) dropped hands fully to attention."
    ))

    # 3. Final Return to Attention
    ends_in_att = bool(segments and segments[-1].get("kind") == "attention")
    items.append(QualityCheckItem(
        check_name="Final Presentation Return to Attention",
        category="presentation",
        passed=ends_in_att,
        status="PASSED" if ends_in_att else "FAILED",
        measured_value="Attention" if ends_in_att else "Non-attention ending",
        threshold="Return to attention at presentation completion",
        frame_range=[0, 0],
        timestamp_range_s=[0.0, round(schedule.get("duration_s", 0.0), 3)],
        detail="Presentation finished smoothly in verified attention pose." if ends_in_att else "Presentation did not return to attention pose at the end."
    ))

    return items
