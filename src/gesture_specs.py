#!/usr/bin/env python3
"""Gesture Specification Registry for Unitree G1.

Defines independent, structured quality specifications for library gestures
derived from their intended prompts. Specifications state expected arm usage,
direction/path, repetitions, hold duration, palm orientation, stationary
body constraints, and presentation allowability.

Ambiguous prompts are explicitly assigned NEEDS_REVIEW status.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, asdict

_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_CUR_DIR) if os.path.basename(_CUR_DIR) == "src" else _CUR_DIR
DATA_DIR = os.path.join(ROOT, "data")

def _data_file(name: str) -> str:
    p = os.path.join(DATA_DIR, name)
    return p if os.path.isfile(p) else os.path.join(ROOT, name)

LIBRARY_250_PATH = _data_file("gesture_library_250.json")


@dataclass
class GestureSpec:
    gesture_id: str
    intent: str
    prompt: str
    arm: str                       # 'right', 'left', 'both', 'none', 'either'
    hand_path: str                 # 'side_to_side', 'upward', 'forward', 'across_chest', 'downward', 'chest_touch', 'open_wide'
    palm_orientation: str          # 'open_forward', 'downward', 'inward_to_chest', 'upward', 'any'
    expected_wave_cycles: int      # 0 for non-repeating, >=1 for cyclic waving/clapping
    min_arm_rom_rad: float         # Minimum range of motion for primary arm joints
    min_hold_seconds: float        # Expected hold time at peak pose
    stationary_parts: list[str] = field(default_factory=lambda: ["legs", "feet", "pelvis"])
    presentation_allowed: bool = True
    spec_status: str = "SPECIFIED"  # 'SPECIFIED', 'NEEDS_REVIEW'
    review_reason: str = ""


# Pre-defined explicit specifications for core gesture library categories
KNOWN_SPECS: dict[str, dict] = {
    # Greetings & Wave
    "PDF_001": {
        "arm": "right", "hand_path": "side_to_side", "palm_orientation": "open_forward",
        "expected_wave_cycles": 2, "min_arm_rom_rad": 0.30, "min_hold_seconds": 0.0,
        "presentation_allowed": True
    },
    "PDF_002": {
        "arm": "right", "hand_path": "upward", "palm_orientation": "open_forward",
        "expected_wave_cycles": 1, "min_arm_rom_rad": 0.25, "min_hold_seconds": 0.2,
        "presentation_allowed": True
    },
    "PDF_003": {
        "arm": "right", "hand_path": "upward", "palm_orientation": "open_forward",
        "expected_wave_cycles": 1, "min_arm_rom_rad": 0.35, "min_hold_seconds": 0.2,
        "presentation_allowed": True
    },
    "PDF_004": {
        "arm": "right", "hand_path": "open_wide", "palm_orientation": "open_forward",
        "expected_wave_cycles": 0, "min_arm_rom_rad": 0.35, "min_hold_seconds": 0.3,
        "presentation_allowed": True
    },
    "PDF_005": {
        "arm": "right", "hand_path": "across_chest", "palm_orientation": "open_forward",
        "expected_wave_cycles": 1, "min_arm_rom_rad": 0.25, "min_hold_seconds": 0.2,
        "presentation_allowed": True
    },
    "PDF_006": {
        "arm": "both", "hand_path": "upward", "palm_orientation": "downward",
        "expected_wave_cycles": 0, "min_arm_rom_rad": 0.25, "min_hold_seconds": 0.3,
        "presentation_allowed": True
    },
    "PDF_007": {
        "arm": "both", "hand_path": "open_wide", "palm_orientation": "open_forward",
        "expected_wave_cycles": 0, "min_arm_rom_rad": 0.35, "min_hold_seconds": 0.4,
        "presentation_allowed": True
    },
    "PDF_010": {
        "arm": "right", "hand_path": "side_to_side", "palm_orientation": "open_forward",
        "expected_wave_cycles": 2, "min_arm_rom_rad": 0.25, "min_hold_seconds": 0.0,
        "presentation_allowed": True
    },
    "PDF_014": {
        "arm": "right", "hand_path": "chest_touch", "palm_orientation": "inward_to_chest",
        "expected_wave_cycles": 0, "min_arm_rom_rad": 0.25, "min_hold_seconds": 0.5,
        "presentation_allowed": True
    },
    "PDF_016": {
        "arm": "both", "hand_path": "forward", "palm_orientation": "upward",
        "expected_wave_cycles": 0, "min_arm_rom_rad": 0.25, "min_hold_seconds": 0.4,
        "presentation_allowed": True
    },
    "PDF_046": {
        "arm": "left", "hand_path": "open_wide", "palm_orientation": "open_forward",
        "expected_wave_cycles": 0, "min_arm_rom_rad": 0.30, "min_hold_seconds": 0.4,
        "presentation_allowed": True
    },
    "PDF_047": {
        "arm": "right", "hand_path": "open_wide", "palm_orientation": "open_forward",
        "expected_wave_cycles": 0, "min_arm_rom_rad": 0.30, "min_hold_seconds": 0.4,
        "presentation_allowed": True
    },
    "PDF_1251": {
        "arm": "right", "hand_path": "side_to_side", "palm_orientation": "open_forward",
        "expected_wave_cycles": 0, "min_arm_rom_rad": 0.25, "min_hold_seconds": 0.3,
        "presentation_allowed": True
    },
    "PDF_1256": {
        "arm": "right", "hand_path": "side_to_side", "palm_orientation": "open_forward",
        "expected_wave_cycles": 0, "min_arm_rom_rad": 0.25, "min_hold_seconds": 0.3,
        "presentation_allowed": True
    },
    "PDF_209": {
        "arm": "both", "hand_path": "forward", "palm_orientation": "upward",
        "expected_wave_cycles": 1, "min_arm_rom_rad": 0.30, "min_hold_seconds": 0.5,
        "presentation_allowed": True
    },

    # Whole-body actions mapped to stationary presentation library gestures
    "PDF_088": {
        "arm": "both", "hand_path": "forward", "palm_orientation": "any",
        "expected_wave_cycles": 0, "min_arm_rom_rad": 0.10, "min_hold_seconds": 0.0,
        "presentation_allowed": True, "spec_status": "SPECIFIED"
    },
    "PDF_090": {
        "arm": "both", "hand_path": "forward", "palm_orientation": "any",
        "expected_wave_cycles": 0, "min_arm_rom_rad": 0.10, "min_hold_seconds": 0.0,
        "presentation_allowed": True, "spec_status": "SPECIFIED"
    },
    "PDF_091": {
        "arm": "both", "hand_path": "forward", "palm_orientation": "any",
        "expected_wave_cycles": 0, "min_arm_rom_rad": 0.10, "min_hold_seconds": 0.0,
        "presentation_allowed": True, "spec_status": "SPECIFIED"
    },
    "PDF_192": {
        "arm": "both", "hand_path": "forward", "palm_orientation": "any",
        "expected_wave_cycles": 0, "min_arm_rom_rad": 0.10, "min_hold_seconds": 0.0,
        "presentation_allowed": True, "spec_status": "SPECIFIED"
    },
    "PDF_193": {
        "arm": "both", "hand_path": "forward", "palm_orientation": "any",
        "expected_wave_cycles": 0, "min_arm_rom_rad": 0.10, "min_hold_seconds": 0.0,
        "presentation_allowed": True, "spec_status": "SPECIFIED"
    },
    "PDF_194": {
        "arm": "both", "hand_path": "forward", "palm_orientation": "any",
        "expected_wave_cycles": 0, "min_arm_rom_rad": 0.10, "min_hold_seconds": 0.0,
        "presentation_allowed": True, "spec_status": "SPECIFIED"
    },
    "PDF_880": {
        "arm": "both", "hand_path": "across_chest", "palm_orientation": "open_forward",
        "expected_wave_cycles": 0, "min_arm_rom_rad": 0.20, "min_hold_seconds": 0.3,
        "presentation_allowed": True, "spec_status": "SPECIFIED"
    },
}

# Ambiguous keyphrases that trigger NEEDS_REVIEW
AMBIGUOUS_KEYWORDS = [
    "think about", "ponder", "contemplate", "feel emotion",
    "philosophical", "daydream", "imagine something vague"
]


def _infer_spec_from_prompt(gesture_id: str, intent: str, prompt: str) -> GestureSpec:
    """Derive specification parameters from prompt text if not in explicit dictionary."""
    p_lower = (prompt + " " + intent).lower()

    # Check ambiguity first
    for kw in AMBIGUOUS_KEYWORDS:
        if kw in p_lower:
            return GestureSpec(
                gesture_id=gesture_id, intent=intent, prompt=prompt,
                arm="either", hand_path="any", palm_orientation="any",
                expected_wave_cycles=0, min_arm_rom_rad=0.15, min_hold_seconds=0.2,
                spec_status="NEEDS_REVIEW",
                review_reason=f"Prompt '{prompt}' contains abstract concept ('{kw}') requiring human intent review."
            )

    # Determine arm usage
    if "both" in p_lower or "two hands" in p_lower or "two palms" in p_lower or "symmetrically" in p_lower or "palms inward" in p_lower:
        arm = "both"
    elif ("right palm" in p_lower or "right hand" in p_lower or "right arm" in p_lower) and not ("left palm" in p_lower or "left hand" in p_lower):
        arm = "right"
    elif ("left palm" in p_lower or "left hand" in p_lower or "left arm" in p_lower) and not ("right palm" in p_lower or "right hand" in p_lower):
        arm = "left"
    elif "right" in p_lower and "left" not in p_lower:
        arm = "right"
    elif "left" in p_lower and "right" not in p_lower:
        arm = "left"
    else:
        arm = "both"  # Default presentation pose assumption

    # Determine hand path
    if "wave" in p_lower or "side to side" in p_lower:
        hand_path = "side_to_side"
    elif "raise" in p_lower or "stretch" in p_lower or "up" in p_lower:
        hand_path = "upward"
    elif "across" in p_lower or "sweep" in p_lower:
        hand_path = "across_chest"
    elif "forward" in p_lower or "point" in p_lower or "extend" in p_lower:
        hand_path = "forward"
    elif "open" in p_lower or "wide" in p_lower:
        hand_path = "open_wide"
    elif "chest" in p_lower or "touch" in p_lower:
        hand_path = "chest_touch"
    else:
        hand_path = "forward"

    # Determine wave cycles
    m_wave = re.search(r"(\d+)\s*time", p_lower)
    if m_wave:
        wave_cycles = int(m_wave.group(1))
    elif "wave" in p_lower or "clap" in p_lower:
        wave_cycles = 2
    else:
        wave_cycles = 0

    # Determine palm orientation
    if "open palm" in p_lower or "forward" in p_lower:
        palm_orientation = "open_forward"
    elif "downward" in p_lower or "lower" in p_lower:
        palm_orientation = "downward"
    elif "upward" in p_lower or "palms up" in p_lower:
        palm_orientation = "upward"
    elif "chest" in p_lower:
        palm_orientation = "inward_to_chest"
    else:
        palm_orientation = "open_forward"

    software_run = bool(re.search(r'\brun\b.*\b(consensus|inference|model|training|code|loop|pipeline|experiment|cycle|round)\b', p_lower)) or any(phrase in p_lower for phrase in ["run inference", "run consensus", "run model", "run training", "run code", "run loop", "run pipeline", "run a consensus", "run one consensus", "rise-over-run", "rise over run", "rate of change"])
    if software_run:
        presentation_allowed = True
    else:
        restricted_keywords = ["walk", "run", "jump", "kick", "flip", "sit", "lie down", "kneel", "crawl", "hop"]
        restricted = any(re.search(r'\b' + re.escape(word) + r'\b', p_lower) for word in restricted_keywords)
        presentation_allowed = not restricted

    return GestureSpec(
        gesture_id=gesture_id, intent=intent, prompt=prompt,
        arm=arm, hand_path=hand_path, palm_orientation=palm_orientation,
        expected_wave_cycles=wave_cycles, min_arm_rom_rad=0.20, min_hold_seconds=0.3,
        presentation_allowed=presentation_allowed, spec_status="SPECIFIED"
    )


def get_gesture_spec(gesture_id: str, intent: str = "", prompt: str = "") -> GestureSpec:
    """Fetch or derive the independent specification for a gesture ID."""
    if gesture_id in KNOWN_SPECS:
        over = KNOWN_SPECS[gesture_id]
        spec = GestureSpec(
            gesture_id=gesture_id,
            intent=intent or gesture_id,
            prompt=prompt or over.get("prompt", ""),
            arm=over["arm"],
            hand_path=over["hand_path"],
            palm_orientation=over["palm_orientation"],
            expected_wave_cycles=over["expected_wave_cycles"],
            min_arm_rom_rad=over["min_arm_rom_rad"],
            min_hold_seconds=over["min_hold_seconds"],
            presentation_allowed=over.get("presentation_allowed", True),
            spec_status=over.get("spec_status", "SPECIFIED"),
            review_reason=over.get("review_reason", "")
        )
        return spec

    # Load from gesture_library_500_1500.json or gesture_library_250.json if prompt/intent not supplied
    if not prompt:
        for json_file in ["gesture_library_1501_2000.json", "gesture_library_500_1500.json", "gesture_library_501_1000.json", "gesture_library_250.json"]:
            path = _data_file(json_file)
            if os.path.exists(path):
                try:
                    with open(path, "r") as f:
                        lib = json.load(f)
                        for g in lib.get("gestures", []):
                            if g.get("gesture_id") == gesture_id or f"PDF_{g.get('number')}" == gesture_id:
                                intent = g.get("intent", intent)
                                prompt = g.get("motion_prompt", prompt)
                                break
                except Exception:
                    pass
                if prompt:
                    break

    return _infer_spec_from_prompt(gesture_id, intent or gesture_id, prompt or intent or gesture_id)
