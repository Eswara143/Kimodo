#!/usr/bin/env python3
"""Fault Injection & Verification Suite for Gesture Quality System.

Runs isolated quality audit checks on deliberately corrupted copies of G1 motions
to verify that the checker detects every specific fault class:
  1. Wrong arm execution
  2. Repetition count mismatch (1 wave vs 2)
  3. Insufficient / flat movement
  4. Discontinuous joint step jump
  5. Base foot movement / instability
  6. Speech/gesture timing alignment lag

Also verifies known-good gestures to confirm zero false positives.
Never modifies production assets.
"""

from __future__ import annotations

import os
import sys
import unittest
import numpy as np

_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_CUR_DIR) if os.path.basename(_CUR_DIR) == "scripts" else _CUR_DIR
SRC_DIR = os.path.join(ROOT, "src")
for _p in (ROOT, SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gesture_specs import get_gesture_spec
from gesture_quality_checker import check_gesture_quality
from gesture_corrector import attempt_gesture_correction, _swap_arm_channels


class TestGestureQualityChecker(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load a standard valid baseline motion (PDF_001)
        csv_path = os.path.join(ROOT, "outputs", "web", "pdf_250_library", "PDF_001", "PDF_001.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Baseline CSV missing at {csv_path}")
        cls.baseline_qpos = np.loadtxt(csv_path, delimiter=",").astype(np.float64)

    def test_01_known_good_passes(self):
        """Verify that a known-good gesture clip passes all kinematic checks without false alarms."""
        spec = get_gesture_spec("PDF_001")
        report = check_gesture_quality(self.baseline_qpos, spec=spec)
        self.assertIn(report.overall_status, ("CHECKS PASSED", "NEEDS REVIEW"))
        self.assertEqual(len(report.failed_checks), 0, f"False positives detected: {report.failed_checks}")

    def test_02_inject_wrong_arm(self):
        """Inject fault: swap arm channels so left arm moves instead of right arm for a right-hand wave."""
        corrupted = _swap_arm_channels(self.baseline_qpos)
        spec = get_gesture_spec("PDF_001")  # Expects right arm
        report = check_gesture_quality(corrupted, spec=spec)

        self.assertEqual(report.overall_status, "FAILED")
        failed_names = [c["check_name"] for c in report.failed_checks]
        self.assertIn("Correct Arm Execution", failed_names)

    def test_03_inject_insufficient_movement(self):
        """Inject fault: flatten arm joint trajectories to a static posture with ROM ~0.01 rad."""
        corrupted = self.baseline_qpos.copy()
        corrupted[:, 22:36] = corrupted[0, 22:36]  # Flatten arm channels
        spec = get_gesture_spec("PDF_001")
        report = check_gesture_quality(corrupted, spec=spec)

        self.assertEqual(report.overall_status, "FAILED")
        failed_names = [c["check_name"] for c in report.failed_checks]
        self.assertIn("Correct Arm Execution", failed_names)

    def test_04_inject_discontinuous_joint_jump(self):
        """Inject fault: add a sudden +1.0 rad step jump at frame 15."""
        corrupted = self.baseline_qpos.copy()
        if len(corrupted) > 20:
            corrupted[15:, 25] += 1.0  # Add step jump to left elbow
        spec = get_gesture_spec("PDF_001")
        report = check_gesture_quality(corrupted, spec=spec)

        self.assertEqual(report.overall_status, "FAILED")
        failed_names = [c["check_name"] for c in report.failed_checks]
        self.assertIn("Time-Based Joint Continuity", failed_names)

    def test_05_inject_foot_movement(self):
        """Inject fault: add 10 cm foot lift / 15 cm slide to pelvic root position."""
        corrupted = self.baseline_qpos.copy()
        t = np.linspace(0, np.pi, len(corrupted))
        corrupted[:, 2] += np.sin(t) * 0.10  # 10 cm Z lift
        corrupted[:, 0] += np.sin(t) * 0.15  # 15 cm X slide
        spec = get_gesture_spec("PDF_001")
        report = check_gesture_quality(corrupted, spec=spec)

        self.assertEqual(report.overall_status, "FAILED")
        failed_names = [c["check_name"] for c in report.failed_checks]
        self.assertIn("Stationary Feet & Base Stability", failed_names)

    def test_06_inject_repetition_mismatch(self):
        """Inject fault: truncate wave gesture to 20 frames so only 1 cycle is performed instead of 2."""
        corrupted = self.baseline_qpos[:20].copy()
        spec = get_gesture_spec("PDF_001")  # Expects 2 wave cycles
        report = check_gesture_quality(corrupted, spec=spec)

        failed_names = [c["check_name"] for c in report.failed_checks]
        self.assertIn("Wave Repetition Count", failed_names)

    def test_07_attempt_auto_correction_success(self):
        """Verify targeted correction engine successfully repairs a wrong-arm execution fault."""
        corrupted = _swap_arm_channels(self.baseline_qpos)
        spec = get_gesture_spec("PDF_001")

        corr_res = attempt_gesture_correction("PDF_001_TEST", corrupted, spec=spec, max_attempts=3)
        self.assertTrue(corr_res.success)
        self.assertEqual(corr_res.final_status, "CHECKS PASSED")
        self.assertTrue(corr_res.promoted)
        self.assertIsNotNone(corr_res.corrected_csv)
        self.assertTrue(os.path.exists(corr_res.corrected_csv))

    def test_08_attempt_auto_correction_max_attempts_ceiling(self):
        """Verify targeted correction engine respects max 3 attempts ceiling on unfixable fault."""
        # Unfixable fault: corrupted non-finite NaN array
        corrupted = self.baseline_qpos.copy()
        corrupted[5:10, :] = np.nan
        spec = get_gesture_spec("PDF_001")

        corr_res = attempt_gesture_correction("PDF_001_NAN_TEST", corrupted, spec=spec, max_attempts=3)
        self.assertFalse(corr_res.success)
        self.assertIn(corr_res.final_status, ("FAILED", "NEEDS REVIEW"))
        self.assertLessEqual(corr_res.attempts_made, 3)

    def test_09_inject_speech_gesture_timing_offset(self):
        """Inject fault: simulate +850 ms timing offset between spoken phrase and gesture main action onset."""
        from gesture_quality_checker import check_presentation_timeline_quality
        mock_schedule = {
            "duration_s": 10.0,
            "gestures": [
                {"gesture_id": "PDF_001", "intent": "Hello", "phrase": "Hello", "main_lag_s": 0.850}
            ],
            "words": [{"text": "Hello", "start_s": 1.0, "end_s": 1.5}],
            "segments": [
                {"kind": "attention", "start_f": 0, "end_f": 15},
                {"kind": "gesture", "start_f": 15, "end_f": 90},
                {"kind": "attention", "start_f": 90, "end_f": 150}
            ]
        }
        items = check_presentation_timeline_quality(mock_schedule)
        failed_items = [it for it in items if not it.passed]
        self.assertGreater(len(failed_items), 0)
        self.assertEqual(failed_items[0].check_name, "Speech/Gesture Main Action Synchronization")
        self.assertIn("+850 ms", failed_items[0].detail)


if __name__ == "__main__":
    unittest.main(verbosity=2)
