#!/usr/bin/env python3
"""Targeted Gesture Correction Engine for Unitree G1.

Diagnoses motion quality faults (arm usage, amplitude, joint step jumps, hold duration)
and applies targeted kinematic fixes.

Enforces a strict limit of at most 3 correction attempts per gesture.
Rigorously validates channel swaps against joint limits, collisions, and palm orientation.
If a swap introduces a collision or limit breach, marks NEEDS REVIEW instead of forcing an unsafe swap.

Preserves original files and saves candidate motions separately.
Promotes candidates to usable library status only after all checks pass.
"""

from __future__ import annotations

import json
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
from gesture_quality_checker import check_gesture_quality, QualityReport

OUT_CORRECTED_DIR = os.path.join(ROOT, "outputs", "corrected_gestures")
RIGHT_SIGN = {"shoulder_pitch": 1, "shoulder_roll": -1, "shoulder_yaw": -1, "elbow": 1,
              "wrist_roll": -1, "wrist_pitch": 1, "wrist_yaw": -1}
ARM_JOINT_ORDER = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"]


@dataclass
class CorrectionAttempt:
    attempt_number: int
    strategy: str
    description: str
    passed_after: bool
    remaining_failures: list[str]


@dataclass
class CorrectionResult:
    gesture_id: str
    original_status: str
    final_status: str
    attempts_made: int
    max_attempts: int
    success: bool
    promoted: bool
    corrected_csv: str | None
    attempt_history: list[dict]
    final_report: dict


def _mirror_arm_channels(qpos: np.ndarray) -> np.ndarray:
    """Mirror active arm to idle arm with right sign reflection."""
    out = qpos.copy()
    left_arm = qpos[:, 22:29].copy()
    right_arm = qpos[:, 29:36].copy()

    left_rom = float(np.max(np.abs(left_arm - left_arm[0])))
    right_rom = float(np.max(np.abs(right_arm - right_arm[0])))

    if left_rom >= right_rom:
        mirrored = left_arm.copy()
        for idx, joint in enumerate(ARM_JOINT_ORDER):
            mirrored[:, idx] *= RIGHT_SIGN[joint]
        out[:, 29:36] = mirrored
    else:
        mirrored = right_arm.copy()
        for idx, joint in enumerate(ARM_JOINT_ORDER):
            mirrored[:, idx] *= RIGHT_SIGN[joint]
        out[:, 22:29] = mirrored
    return out


def _swap_arm_channels(qpos: np.ndarray) -> np.ndarray:
    """Swap left arm (22:29) and right arm (29:36) channels with right sign mirroring."""
    out = qpos.copy()
    left_arm = qpos[:, 22:29].copy()
    right_arm = qpos[:, 29:36].copy()

    mirrored_right_to_left = right_arm.copy()
    mirrored_left_to_right = left_arm.copy()
    for idx, joint in enumerate(ARM_JOINT_ORDER):
        mirrored_right_to_left[:, idx] *= RIGHT_SIGN[joint]
        mirrored_left_to_right[:, idx] *= RIGHT_SIGN[joint]

    out[:, 22:29] = mirrored_right_to_left
    out[:, 29:36] = mirrored_left_to_right
    return out


def _scale_arm_amplitude(qpos: np.ndarray, target_arm: str, scale: float = 1.35) -> np.ndarray:
    """Scale arm joint excursions relative to base pose to amplify motion."""
    out = qpos.copy()
    base_pose = qpos[0]
    if target_arm in ("right", "both"):
        sl = slice(29, 36)
        out[:, sl] = base_pose[sl] + (qpos[:, sl] - base_pose[sl]) * scale
    if target_arm in ("left", "both"):
        sl = slice(22, 29)
        out[:, sl] = base_pose[sl] + (qpos[:, sl] - base_pose[sl]) * scale
    return out


def _smooth_joint_jumps(qpos: np.ndarray, max_step_rad: float = 0.35) -> np.ndarray:
    """Smooth sudden single-frame joint step jumps using moving average interpolation."""
    out = qpos.copy()
    hinges = out[:, 7:36]
    for c in range(hinges.shape[1]):
        col = hinges[:, c]
        diffs = np.abs(np.diff(col))
        for i in range(len(diffs)):
            if diffs[i] > max_step_rad:
                prev_val = col[max(0, i - 1)]
                next_val = col[min(len(col) - 1, i + 2)]
                col[i + 1] = 0.5 * (prev_val + next_val)
        hinges[:, c] = col
    out[:, 7:36] = hinges
    return out


def _extend_peak_hold(qpos: np.ndarray, hold_frames: int = 12) -> np.ndarray:
    """Duplicate/hold peak posture frame to increase gesture hold duration."""
    n = len(qpos)
    if n < 10:
        return qpos.copy()
    # Find frame of maximum arm displacement from rest
    arms = qpos[:, 22:36]
    dev = np.linalg.norm(arms - arms[0], axis=1)
    peak_idx = int(np.argmax(dev))
    peak_frame = qpos[peak_idx:peak_idx+1]
    hold_block = np.repeat(peak_frame, hold_frames, axis=0)
    out = np.concatenate([qpos[:peak_idx], hold_block, qpos[peak_idx:]], axis=0)
    return out


def _resolve_self_collision(qpos: np.ndarray, offset_rad: float = 0.28) -> np.ndarray:
    """Adjust arm shoulder roll/yaw/pitch and elbow joints outward to eliminate self-collision."""
    out = qpos.copy()
    # Left arm: shoulder_pitch (idx 22), shoulder_roll (idx 23), shoulder_yaw (idx 24), elbow (idx 25)
    out[:, 22] = np.clip(out[:, 22] - 0.20, -1.5, 0.0)
    out[:, 23] = np.clip(out[:, 23] + offset_rad, 0.20, 1.20)
    out[:, 24] += 0.15
    out[:, 25] = -0.50
    out[:, 26:29] = 0.0  # Reset wrist roll/pitch/yaw to neutral

    # Right arm: shoulder_pitch (idx 29), shoulder_roll (idx 30), shoulder_yaw (idx 31), elbow (idx 32)
    out[:, 29] = np.clip(out[:, 29] - 0.20, -1.5, 0.0)
    out[:, 30] = np.clip(out[:, 30] - offset_rad, -1.20, -0.20)
    out[:, 31] -= 0.15
    out[:, 32] = 0.50
    out[:, 33:36] = 0.0  # Reset wrist roll/pitch/yaw to neutral

    # Lock pelvis, legs, and torso (0:22) to stationary posture
    out[:, :22] = 0.0
    return out


def _synthesize_clean_arm_action(qpos: np.ndarray, target_arm: str = "both", amplitude: float = 0.45) -> np.ndarray:
    """Synthesize clean, kinematically smooth presentation gesture trajectory on target arm."""
    n = max(len(qpos), 90)
    out = np.zeros((n, 36))
    t = np.linspace(0, np.pi, n)
    sine_wave = amplitude * np.sin(t)

    if target_arm in ("right", "both"):
        out[:, 29] = -sine_wave         # Right shoulder pitch
        out[:, 30] = -0.30 * sine_wave  # Right shoulder roll outward
        out[:, 32] = 0.60 * sine_wave   # Right elbow bend

    if target_arm in ("left", "both"):
        out[:, 22] = -sine_wave         # Left shoulder pitch
        out[:, 23] = 0.30 * sine_wave   # Left shoulder roll outward
        out[:, 25] = -0.60 * sine_wave  # Left elbow bend

    return out


def _dampen_velocity_acceleration(qpos: np.ndarray, factor: float = 1.3) -> np.ndarray:
    """Interpolate frames temporally to reduce joint velocity and acceleration below ceilings."""
    n_orig = len(qpos)
    n_new = int(round(n_orig * factor))
    orig_times = np.linspace(0, 1, n_orig)
    new_times = np.linspace(0, 1, n_new)

    out = np.zeros((n_new, qpos.shape[1]))
    for col in range(qpos.shape[1]):
        out[:, col] = np.interp(new_times, orig_times, qpos[:, col])

    # Apply 3-point smoothing across time
    hinges = out[:, 7:36]
    for c in range(hinges.shape[1]):
        hinges[:, c] = np.convolve(hinges[:, c], [0.2, 0.6, 0.2], mode="same")
    out[:, 7:36] = hinges
    return out


def attempt_gesture_correction(gesture_id: str, original_qpos: np.ndarray,
                               spec: GestureSpec | None = None,
                               max_attempts: int = 3,
                               fps: float = 30.0) -> CorrectionResult:
    """Diagnose and perform targeted corrections on a gesture motion up to max_attempts."""
    original_qpos = np.asarray(original_qpos, dtype=np.float64)
    if spec is None:
        spec = get_gesture_spec(gesture_id)

    initial_report = check_gesture_quality(original_qpos, spec=spec, fps=fps)
    if initial_report.overall_status in ("CHECKS PASSED", "BLOCKED"):
        return CorrectionResult(
            gesture_id=gesture_id,
            original_status=initial_report.overall_status,
            final_status=initial_report.overall_status,
            attempts_made=0,
            max_attempts=max_attempts,
            success=initial_report.overall_status == "CHECKS PASSED",
            promoted=initial_report.overall_status == "CHECKS PASSED",
            corrected_csv=None,
            attempt_history=[],
            final_report=asdict(initial_report)
        )

    current_qpos = original_qpos.copy()
    current_report = initial_report
    attempt_history: list[dict] = []

    out_dir = os.path.join(OUT_CORRECTED_DIR, gesture_id)
    os.makedirs(out_dir, exist_ok=True)

    for attempt in range(1, max_attempts + 1):
        failed_names = [c["check_name"] for c in current_report.failed_checks]
        candidate_qpos = current_qpos.copy()
        strategy = ""
        desc = ""

        if "Correct Arm Execution" in failed_names:
            if spec.arm == "both":
                strategy = "MIRROR_ARM_CHANNELS"
                desc = "Mirroring active arm to idle arm for two-handed gesture execution."
                candidate_qpos = _mirror_arm_channels(candidate_qpos)
            else:
                strategy = "SWAP_ARM_CHANNELS"
                desc = f"Attempting arm channel swap & sign mirroring to target {spec.arm.upper()} arm execution."
                candidate_qpos = _swap_arm_channels(candidate_qpos)
        elif "Smooth Joint Continuity (No Jumps)" in failed_names or "Time-Based Joint Continuity" in failed_names:
            strategy = "SMOOTH_JOINT_JUMPS"
            desc = "Applied trajectory moving-window smoothing to eliminate step jumps."
            candidate_qpos = _smooth_joint_jumps(candidate_qpos)
        elif "Gesture Hold Duration" in failed_names:
            strategy = "EXTEND_PEAK_HOLD"
            desc = f"Inserted {int(spec.min_hold_seconds * fps)} peak hold frames at posture apex."
            candidate_qpos = _extend_peak_hold(candidate_qpos, hold_frames=int(spec.min_hold_seconds * fps))
        else:
            strategy = "SCALE_ARM_AMPLITUDE"
            desc = f"Amplified arm range of motion by 1.35x for {spec.arm} arm."
            candidate_qpos = _scale_arm_amplitude(candidate_qpos, spec.arm, scale=1.35)

        # Rigorous post-correction validation (verify swap doesn't induce collisions or joint breaches)
        cand_report = check_gesture_quality(candidate_qpos, spec=spec, fps=fps)

        # Safety & physical clearance guard: if candidate introduces self-collision excess > 0 or joint limit breach, reject
        has_new_safety_fault = any(c["check_name"] in ("Self-Collision & Clearance", "Hardware Joint Limits") and not c["passed"] for c in cand_report.failed_checks)

        if has_new_safety_fault and strategy == "SWAP_ARM_CHANNELS":
            # Channel swap created a collision or limit breach: abort swap and mark NEEDS REVIEW
            desc += " [ABORTED: Channel swap induced self-collision or joint limit breach; marked NEEDS REVIEW]"
            cand_report.overall_status = "NEEDS REVIEW"
            passed_now = False
        else:
            passed_now = cand_report.overall_status == "CHECKS PASSED"
            current_qpos = candidate_qpos

        remaining_fails = [c["check_name"] for c in cand_report.failed_checks]

        att_record = CorrectionAttempt(
            attempt_number=attempt,
            strategy=strategy,
            description=desc,
            passed_after=passed_now,
            remaining_failures=remaining_fails
        )
        attempt_history.append(asdict(att_record))
        current_report = cand_report

        if passed_now:
            break

    final_success = current_report.overall_status == "CHECKS PASSED"
    corrected_csv_path = None

    if final_success:
        corrected_csv_path = os.path.join(out_dir, f"{gesture_id}_corrected.csv")
        np.savetxt(corrected_csv_path, current_qpos, delimiter=",", fmt="%.6f")

    return CorrectionResult(
        gesture_id=gesture_id,
        original_status=initial_report.overall_status,
        final_status=current_report.overall_status,
        attempts_made=len(attempt_history),
        max_attempts=max_attempts,
        success=final_success,
        promoted=final_success,
        corrected_csv=corrected_csv_path,
        attempt_history=attempt_history,
        final_report=asdict(current_report)
    )
