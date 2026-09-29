#!/usr/bin/env python3
"""Compose a sentence into a timed sequence of already generated G1 gestures.

This retrieval-and-composition step never runs Kimodo diffusion.  It finds exact
phrases from ``gesture_library.json``, joins the corresponding validated qpos
motions with smooth idle transitions, then writes one combined NPZ archive.

The archive's primary field is ``qpos`` with shape ``(T, 36)`` at 30 fps.  It
also embeds ``timeline_json`` with the planned speech/gesture alignment.  A TTS
player should start audio at ``speech_offset_s`` and honour event pauses.

    python compose_sentence_gestures.py \
      "Hello everyone, thank you for coming. Please be careful." \
      --output outputs/sentence_sequences/example.npz

Simulation-side only; this is not a physical-robot control command.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass

import numpy as np


_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_CUR_DIR) if os.path.basename(_CUR_DIR) == "src" else _CUR_DIR
DATA_DIR = os.path.join(ROOT, "data")

def _data_file(name: str) -> str:
    p = os.path.join(DATA_DIR, name)
    return p if os.path.isfile(p) else os.path.join(ROOT, name)

LIBRARY = _data_file("gesture_library.json")
BOOK = _data_file("gesture_book.json")
LIBRARY_OUTPUT = os.path.join(ROOT, "outputs", "gesture_library")
FPS = 30.0


@dataclass(frozen=True)
class Match:
    gesture_id: str
    intent: str
    phrase: str
    word_start: int
    word_end: int


def _word_count(text: str) -> int:
    return len(re.findall(r"\w+", text, flags=re.UNICODE))


def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    escaped = re.escape(phrase.strip()).replace(r"\ ", r"\s+")
    return re.compile(rf"(?<!\w){escaped}(?!\w)", re.IGNORECASE | re.UNICODE)


LIBRARY_250 = _data_file("gesture_library_250.json")
LIBRARY_250_OUTPUT = os.path.join(ROOT, "outputs", "web", "pdf_250_library")


def _phrases_for(intent: str) -> list[str]:
    """Sentence text that should trigger a gesture.

    The library names each gesture by the word or phrase it stands for.
    This function expands the base intent to catch common spoken variants,
    synonyms, and verb forms so natural sentence inputs match perfectly.
    """
    base = intent.strip().lower()
    variants = {base}
    
    # Rich synonym & spoken variant dictionary covering key library categories
    extra = {
        # Greetings & Polite
        "hello": {"hello", "hi", "hey", "hello everyone", "hi everyone", "greetings"},
        "hi": {"hi", "hey"},
        "hey": {"hey"},
        "good morning": {"good morning", "morning"},
        "good afternoon": {"good afternoon", "afternoon"},
        "good evening": {"good evening", "evening"},
        "welcome": {"welcome", "welcome to", "welcome everyone", "you are welcome"},
        "namaste": {"namaste", "namaskar"},
        "goodbye": {"goodbye", "good bye", "bye", "see you"},
        "bye": {"bye", "bye bye"},
        "thank you": {"thank you", "thanks", "thank you very much", "many thanks"},
        "thanks": {"thanks", "thank you"},
        "please": {"please", "kindly"},
        "sorry": {"sorry", "i am sorry", "i'm sorry", "my apologies"},
        "congratulations": {"congratulations", "congrats", "congratulation"},
        "nice to meet you": {"nice to meet you", "pleased to meet you"},
        "come again": {"come again", "visit again"},
        
        # Answers & Conversation
        "yes": {"yes", "yeah", "yep", "sure"},
        "no": {"no", "nope", "not at all"},
        "okay": {"okay", "ok", "alright"},
        "correct": {"correct", "that's right", "exact"},
        "wrong": {"wrong", "incorrect", "false"},
        "maybe": {"maybe", "perhaps"},
        "i don't know": {"i don't know", "i dont know", "no idea"},
        "i know": {"i know", "i am aware"},
        "understand": {"understand", "got it", "comprehend"},
        "i understand": {"i understand", "i see"},
        "i don't understand": {"i don't understand", "i dont understand", "confused"},
        "explain": {"explain", "let me explain", "explanation", "explaining"},
        "speak": {"speak", "speaking", "talk", "talking"},
        "listen": {"listen", "listening", "hear"},
        "quiet": {"quiet", "be quiet", "silence"},
        "wait": {"wait", "hold on", "wait a moment"},
        "repeat": {"repeat", "say again"},
        
        # Directions & Navigation
        "left": {"left", "to the left"},
        "right": {"right", "to the right"},
        "forward": {"forward", "straight ahead"},
        "backward": {"backward", "backwards", "back"},
        "up": {"up", "upward"},
        "down": {"down", "downward"},
        "here": {"here", "right here"},
        "there": {"there", "over there"},
        "turn left": {"turn left", "turning left"},
        "turn right": {"turn right", "turning right"},
        "go forward": {"go forward", "move forward"},
        "go back": {"go back", "step back"},
        "come here": {"come here", "come over"},
        "follow me": {"follow me"},
        "this way": {"this way"},
        "inside": {"inside", "enter"},
        "outside": {"outside", "exterior"},
        
        # Commands & Safety
        "stop": {"stop", "halt", "freeze"},
        "go": {"go", "proceed"},
        "start": {"start", "begin", "commence"},
        "finish": {"finish", "done", "complete", "completed"},
        "continue": {"continue", "keep going"},
        "pause": {"pause", "take a pause"},
        "slow": {"slow", "slowly", "slow down"},
        "fast": {"fast", "quickly", "speed up"},
        "stay": {"stay", "remain"},
        "danger": {"danger", "dangerous"},
        "warning": {"warning", "caution"},
        "emergency": {"emergency"},
        "help": {"help", "help me", "assist", "assistance"},
        "careful": {"careful", "be careful", "stay safe"},
        "look": {"look", "look at"},
        "show": {"show", "show me", "demonstrate", "demonstration"},
        "study": {"study", "studying"},
        "learn": {"learn", "learning"},
        "work": {"work", "working", "our project", "project"},
        "robot": {"robot", "robotics", "robotics lab"},

        # Shapes & Visual
        "circle": {"circle", "circular", "round"},
        "triangle": {"triangle", "triangular"},
        "rectangle": {"rectangle", "rectangular"},
        "zigzag": {"zigzag", "zig-zag"},
        "spiral": {"spiral"},
        "arch": {"arch"},
        "heart shape": {"heart shape", "heart"},
        "star shape": {"star shape", "star"},
        "oval": {"oval"},
        "wavy line": {"wavy line", "wavy"},

        # Dimensions & Quantity
        "tall": {"tall", "height", "high"},
        "short height": {"short height", "short"},
        "wide": {"wide", "broad"},
        "narrow": {"narrow"},
        "thick": {"thick"},
        "thin": {"thin"},
        "equal amounts": {"equal amounts", "equal"},
        "unequal amounts": {"unequal amounts", "unequal"},

        # Devices & Media
        "zoom in": {"zoom in", "zooming in"},
        "zoom out": {"zoom out", "zooming out"},
        "swipe a card": {"swipe a card", "swipe card"},
        "scan a barcode": {"scan a barcode", "scan barcode"},
        "drag an item": {"drag an item", "drag item"},
        "resize a window": {"resize a window", "resize window"},
        "scroll a page": {"scroll a page", "scroll page"},
        "mute a speaker": {"mute a speaker", "mute speaker", "mute"},

        # Music & Leisure
        "play a violin": {"play a violin", "violin"},
        "play a guitar": {"play a guitar", "guitar"},
        "play a piano": {"play a piano", "piano"},
        "play drums": {"play drums", "drums"},
        "conduct music": {"conduct music", "conducting"},

        # Tasks & Utilities
        "pour": {"pour", "pouring"},
        "sprinkle": {"sprinkle", "sprinkling"},
        "fasten a seat belt": {"fasten a seat belt", "fasten seatbelt", "seatbelt"},
        "orbit": {"orbit", "orbiting"},
        "pendulum": {"pendulum"},
        "expansion": {"expansion", "expanding"},
        "compression": {"compression", "compressing"},
        "build a tower": {"build a tower", "building a tower"},
        "staple papers": {"staple papers", "stapling"},
    }
    
    variants |= extra.get(base, set())
    return sorted(variants, key=len, reverse=True)


def load_catalog() -> list[dict]:
    """Load catalog gestures that have a generated, structurally valid CSV."""
    catalog = []

    for lib_json in ["gesture_library_1501_2000.json", "gesture_library_500_1500.json", "gesture_library_501_1000.json", "gesture_library_250.json"]:
        lib_path = _data_file(lib_json)
        if os.path.isfile(lib_path) and os.path.isdir(LIBRARY_250_OUTPUT):
            for g in json.load(open(lib_path)).get("gestures", []):
                gid = g.get("gesture_id") or f"PDF_{g.get('number')}"
                csv_path = os.path.join(LIBRARY_250_OUTPUT, gid, f"{gid}.csv")
                if not os.path.isfile(csv_path):
                    continue          # not generated yet, or it failed verification
                entry = dict(g)
                entry.update({"source_csv": csv_path,
                              "example_phrases": _phrases_for(g["intent"]),
                              "validation_warnings": []})
                catalog.append(entry)
            if catalog:
                return catalog

    if os.path.isfile(LIBRARY) and os.path.isfile(BOOK):
        definitions = {g["gesture_id"]: g for g in json.load(open(LIBRARY))["gestures"]}
        book = json.load(open(BOOK))
        for generated in book.get("gestures", []):
            gid = generated.get("gesture_id")
            csv_path = os.path.join(LIBRARY_OUTPUT, str(gid), f"{gid}.csv")
            if gid in definitions and generated.get("validation_status") == "generated" and os.path.isfile(csv_path):
                entry = dict(definitions[gid])
                entry.update({
                    "source_csv": csv_path,
                    "validation_warnings": generated.get("validation_warnings", []),
                })
                catalog.append(entry)

    if not catalog:
        raise RuntimeError("No generated gesture-library CSV assets were found.")
    return catalog


HIGH_PRIORITY_INTENTS = {
    # Core Greetings
    "hello", "hi", "hey", "welcome", "goodbye", "bye", "namaste", "thank you", "thanks", "nice to meet you",
    "sorry", "please", "congratulations", "good morning", "good afternoon", "good evening",
    # Core Answers & Navigation
    "yes", "no", "okay", "correct", "wrong", "stop", "go", "left", "right", "forward", "backward", "up", "down",
    "turn left", "turn right", "here", "there", "start", "finish", "continue", "pause",
    # Core Actions & Presentation
    "explain", "demonstrate", "present", "show", "study", "learn", "teach", "question", "answer",
    "danger", "warning", "emergency", "help", "stay back", "careful", "be careful", "fire",
    "eat", "drink", "sleep", "wake up", "call", "phone", "work", "carry", "throw", "catch", "walk", "run", "dance",
    # Visual & Devices
    "circle", "triangle", "rectangle", "tall", "short height", "wide", "zoom in", "zoom out",
    "pour", "play a violin", "play a guitar", "play a piano", "play drums"
}


def intent_priority(intent: str) -> int:
    """Return priority level (1=High Impact Key Intent, 2=Secondary Intent)."""
    intent_clean = intent.strip().lower()
    if intent_clean in HIGH_PRIORITY_INTENTS:
        return 1
    return 2


def find_matches(sentence: str, catalog: list[dict],
                 max_gestures: int | None = None) -> list[Match]:
    """Find key phrase matches, prioritizing high-impact intents and spacing gestures naturally."""
    candidates: list[tuple[int, int, int, int, dict, str]] = []
    for order, gesture in enumerate(catalog):
        intent = gesture.get("intent", "")
        prio = intent_priority(intent)
        for phrase in gesture.get("example_phrases", []):
            for found in _phrase_pattern(phrase).finditer(sentence):
                candidates.append((found.start(), prio, -(found.end() - found.start()), order, gesture, found.group(0)))
                
    # Sort by character start position, then by priority level, then by match length
    candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))

    total_words = _word_count(sentence)
    
    # Calculate intelligent gesture cap based on sentence/paragraph length
    if max_gestures is None:
        if total_words <= 15:
            dynamic_cap = 3
        elif total_words <= 40:
            dynamic_cap = 4
        else:
            dynamic_cap = 6
    else:
        dynamic_cap = max_gestures

    matches, used_spans = [], []
    last_word_idx = -10

    for start, prio, _neg_len, _order, gesture, phrase in candidates:
        end = start + len(phrase)
        word_start = _word_count(sentence[:start])
        word_end = word_start + _word_count(phrase)
        
        # Avoid character overlap
        if any(start < occupied_end and end > occupied_start for occupied_start, occupied_end in used_spans):
            continue
            
        # Avoid crowded back-to-back gestures unless high priority
        if word_start - last_word_idx < 4 and prio > 1:
            continue

        matches.append(Match(gesture["gesture_id"], gesture["intent"], phrase, word_start, word_end))
        used_spans.append((start, end))
        last_word_idx = word_end
        
        if len(matches) >= dynamic_cap:
            break
            
    matches.sort(key=lambda m: m.word_start)
    return matches


def _norm_quat(q: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        raise ValueError("Cannot use a zero-length root quaternion")
    return q / norm


def _slerp_wxyz(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    a, b = _norm_quat(a), _norm_quat(b)
    dot = float(np.dot(a, b))
    if dot < 0:
        b, dot = -b, -dot
    if dot > 0.9995:
        return _norm_quat((1 - t) * a + t * b)
    theta = float(np.arccos(np.clip(dot, -1, 1)))
    return (np.sin((1 - t) * theta) * a + np.sin(t * theta) * b) / np.sin(theta)


def blend_qpos(start: np.ndarray, end: np.ndarray, frames: int) -> np.ndarray:
    """Smoothly interpolate a free-joint qpos pose, including quaternion SLERP."""
    out = np.empty((frames, len(start)), dtype=np.float64)
    for i in range(frames):
        t = (i + 1) / frames
        out[i] = (1 - t) * start + t * end
        out[i, 3:7] = _slerp_wxyz(start[3:7], end[3:7], t)
    return out


def _idle_pose(catalog: list[dict]) -> np.ndarray:
    """A known valid standing frame: the first frame of the welcome gesture."""
    source = next((item for item in catalog if item["gesture_id"] == "G003"), catalog[0])
    pose = np.loadtxt(source["source_csv"], delimiter=",")[0].astype(np.float64)
    pose[3:7] = _norm_quat(pose[3:7])
    return pose


def resample_clip(clip: np.ndarray, target_frames: int) -> np.ndarray:
    """Resample a qpos gesture clip smoothly to target_frames."""
    if len(clip) == target_frames or target_frames <= 5:
        return clip
    old_indices = np.linspace(0, len(clip) - 1, num=len(clip))
    new_indices = np.linspace(0, len(clip) - 1, num=target_frames)
    resampled = np.empty((target_frames, clip.shape[1]), dtype=clip.dtype)
    for col in range(clip.shape[1]):
        resampled[:, col] = np.interp(new_indices, old_indices, clip[:, col])
    return resampled


def _talking_pose_base(idle: np.ndarray) -> np.ndarray:
    """A comfortable, natural posture at stomach/lower-chest level with forearms raised ~10-15 deg and elbows bent."""
    pose = idle.copy()
    # Left arm: shoulder pitch forward ~0.35, roll outward ~0.18, elbow bent ~0.55, wrist roll ~0.15
    pose[22] = 0.35   # left_shoulder_pitch
    pose[23] = 0.18   # left_shoulder_roll
    pose[24] = 0.10   # left_shoulder_yaw
    pose[25] = 0.55   # left_elbow (flexed more to raise forearm)
    pose[26] = 0.15   # left_wrist_roll
    
    # Right arm: shoulder pitch forward ~0.35, roll outward ~ -0.18, elbow bent ~0.55, wrist roll ~ -0.15
    pose[29] = 0.35   # right_shoulder_pitch
    pose[30] = -0.18  # right_shoulder_roll
    pose[31] = -0.10  # right_shoulder_yaw
    pose[32] = 0.55   # right_elbow (flexed more to raise forearm)
    pose[33] = -0.15  # right_wrist_roll
    return pose


def intercept_clip_to_talking_pose(clip: np.ndarray, target_pose: np.ndarray, return_fraction: float = 0.30) -> np.ndarray:
    """Modify the trailing return phase of a gesture clip so hands settle at stomach-level talking pose instead of dropping to side hips."""
    out = clip.copy().astype(np.float64)
    n = len(out)
    if n <= 10:
        return out
    m = max(1, int(n * (1.0 - return_fraction)))
    rem = n - m
    for i in range(rem):
        alpha = (i + 1) / rem
        s_alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        idx = m + i
        out[idx] = (1.0 - s_alpha) * out[idx] + s_alpha * target_pose
        out[idx, 3:7] = _slerp_wxyz(out[idx, 3:7], target_pose[3:7], s_alpha)
    return out


def generate_talking_idle(start_pose: np.ndarray, num_frames: int, catalog: list[dict],
                          global_frame_offset: int = 0) -> np.ndarray:
    """Generate a smooth talking idle sequence with organic micro-movements at stomach level."""
    idle = _idle_pose(catalog)
    target_base = _talking_pose_base(idle)
    frames = np.empty((num_frames, len(start_pose)), dtype=np.float64)
    
    blend_in = min(15, num_frames)
    for i in range(num_frames):
        t_global = (global_frame_offset + i) / FPS
        # Organic micro-movements with +25% range scaling for visible casual talking motions
        micro_l_elbow = np.sin(2 * np.pi * 0.45 * t_global) * 0.05
        micro_r_elbow = np.cos(2 * np.pi * 0.38 * t_global + 0.4) * 0.05
        micro_l_pitch = np.sin(2 * np.pi * 0.30 * t_global + 0.5) * 0.031
        micro_r_pitch = np.cos(2 * np.pi * 0.26 * t_global + 0.3) * 0.031
        micro_l_wrist = np.sin(2 * np.pi * 0.50 * t_global + 0.2) * 0.0375
        micro_r_wrist = np.cos(2 * np.pi * 0.42 * t_global + 0.7) * 0.0375
        micro_waist_pitch = np.sin(2 * np.pi * 0.22 * t_global) * 0.0185

        pose = target_base.copy()
        pose[21] += micro_waist_pitch   # waist_pitch
        pose[22] += micro_l_pitch       # left_shoulder_pitch
        pose[25] += micro_l_elbow       # left_elbow
        pose[26] += micro_l_wrist       # left_wrist_roll
        pose[29] += micro_r_pitch       # right_shoulder_pitch
        pose[32] += micro_r_elbow       # right_elbow
        pose[33] += micro_r_wrist       # right_wrist_roll

        if i < blend_in:
            alpha = (i + 1) / blend_in
            s_alpha = alpha * alpha * (3.0 - 2.0 * alpha)
            frames[i] = (1.0 - s_alpha) * start_pose + s_alpha * pose
            frames[i, 3:7] = _slerp_wxyz(start_pose[3:7], pose[3:7], s_alpha)
        else:
            frames[i] = pose
            
    return frames


def compose(sentence: str, catalog: list[dict], matches: list[Match], *,
            words_per_minute: float = 150.0, blend_seconds: float = 0.3,
            tail_seconds: float = 0.3,
            speech_duration_s: float | None = None) -> tuple[np.ndarray, dict]:
    """Return a composed G1 qpos timeline with direct gesture continuation synchronized to speech duration."""
    if words_per_minute <= 0 or blend_seconds < 0 or tail_seconds < 0:
        raise ValueError("Speech speed must be positive and durations cannot be negative")
    by_id = {item["gesture_id"]: item for item in catalog}
    seconds_per_word = 60.0 / words_per_minute
    words = max(_word_count(sentence), 1)
    blend_frames = int(round(blend_seconds * FPS))
    speech_offset_frames = blend_frames if matches else 0
    idle = _idle_pose(catalog)
    pieces: list[np.ndarray] = []
    events: list[dict] = []
    cursor, speech_cursor, previous_word_end = 0, speech_offset_frames / FPS, 0

    talking_base = _talking_pose_base(idle)
    current_pose = talking_base

    # Calculate target gesture clip duration if speech_duration_s is provided
    target_clip_frames = None
    if speech_duration_s and speech_duration_s > 0 and matches:
        avail_speech_s = max(1.5, speech_duration_s - (blend_seconds * (len(matches) + 1)))
        target_clip_frames = max(30, int(round((avail_speech_s / len(matches)) * FPS)))

    for match in matches:
        item = by_id[match.gesture_id]
        clip = np.loadtxt(item["source_csv"], delimiter=",")
        if clip.ndim == 1:
            clip = clip[None, :]
        if clip.shape[1] != 36:
            raise ValueError(f"{match.gesture_id} has {clip.shape[1]} qpos columns; expected 36")

        if target_clip_frames:
            clip = resample_clip(clip, target_clip_frames)

        clip = intercept_clip_to_talking_pose(clip, talking_base)

        speech_cursor += (match.word_start - previous_word_end) * seconds_per_word
        unpaused_start = speech_cursor

        # Direct blend from current pose to the start of the next gesture clip
        pieces.append(blend_qpos(current_pose, clip[0], blend_frames))
        cursor += blend_frames
        speech_start = cursor / FPS
        speech_cursor = max(speech_cursor, speech_start)

        # Append gesture motion clip
        pieces.append(clip.astype(np.float64))
        cursor += len(clip)

        # Update current pose to the end frame of this gesture for next transition
        current_pose = clip[-1]
        speech_cursor += (match.word_end - match.word_start) * seconds_per_word
        previous_word_end = match.word_end

        events.append({
            "gesture_id": match.gesture_id,
            "intent": match.intent,
            "matched_phrase": match.phrase,
            "word_span": [match.word_start, match.word_end],
            "speech_start_s": round(speech_start, 3),
            "pause_before_s": round(max(0, speech_start - unpaused_start), 3),
            "motion_end_s": round(cursor / FPS, 3),
            "source_csv": os.path.relpath(item["source_csv"], ROOT),
            "source_validation_warnings": item["validation_warnings"],
        })

    # Maintain stomach talking idle pose at the end of the sentence
    if matches:
        tail_idle = generate_talking_idle(current_pose, blend_frames * 2, catalog, global_frame_offset=cursor)
        pieces.append(tail_idle)
        cursor += len(tail_idle)
    else:
        tail_idle = generate_talking_idle(talking_base, int(round(3.0 * FPS)), catalog, global_frame_offset=0)
        pieces.append(tail_idle)

    speech_cursor += (words - previous_word_end) * seconds_per_word
    qpos = np.concatenate(pieces, axis=0) if pieces else talking_base[None, :]
    return qpos, {
        "format": "kimodo-g1-sentence-sequence-v1",
        "sentence": sentence,
        "fps": FPS,
        "frames": int(len(qpos)),
        "duration_s": round(len(qpos) / FPS, 3),
        "speech_offset_s": round(speech_offset_frames / FPS, 3),
        "scheduled_speech_end_s": round(speech_cursor, 3),
        "words_per_minute": words_per_minute,
        "blend_seconds": round(blend_frames / FPS, 3),
        "gestures": events,
        "no_gesture_match": not events,
    }


def validate(qpos: np.ndarray) -> tuple[bool, list[str], str]:
    """Run the existing validator via a temporary CSV; only the NPZ is retained."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as handle:
        csv_path = handle.name
    try:
        np.savetxt(csv_path, qpos, delimiter=",")
        proc = subprocess.run([sys.executable, os.path.join(ROOT, "validate_g1_motion.py"), csv_path],
                              cwd=ROOT, capture_output=True, text=True)
    finally:
        if os.path.exists(csv_path):
            os.unlink(csv_path)
    report = proc.stdout + ("\n" + proc.stderr if proc.stderr else "")
    warnings = [line.strip() for line in report.splitlines() if line.startswith("WARNING:")]
    return proc.returncode == 0, warnings, report


def write_npz(output: str, qpos: np.ndarray, metadata: dict) -> str:
    """Write the final sequence artifact and no sidecar CSV/JSON files."""
    output = os.path.abspath(output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    np.savez_compressed(
        output,
        qpos=qpos.astype(np.float32),
        sentence=np.asarray(metadata["sentence"]),
        fps=np.asarray(FPS, dtype=np.float32),
        speech_offset_s=np.asarray(metadata["speech_offset_s"], dtype=np.float32),
        gesture_ids=np.asarray([event["gesture_id"] for event in metadata["gestures"]], dtype="U8"),
        timeline_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    return output


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sentence")
    ap.add_argument("--output", default="outputs/sentence_sequences/sequence.npz")
    ap.add_argument("--words-per-minute", type=float, default=150.0)
    ap.add_argument("--max-gestures", type=int, default=None,
                    help="cap the gesture count (default: every matched phrase)")
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()
    if not args.sentence.strip() or (args.max_gestures is not None and args.max_gestures < 1):
        ap.error("provide a non-empty sentence and --max-gestures >= 1")

    catalog = load_catalog()
    matches = find_matches(args.sentence, catalog, args.max_gestures)
    if args.plan_only:
        for match in matches:
            print(f"{match.gesture_id:4s} {match.intent:16s} <- {match.phrase!r}")
        if not matches:
            print("No catalog phrase matched; the result would be idle motion only.")
        return 0
    qpos, metadata = compose(args.sentence, catalog, matches, words_per_minute=args.words_per_minute)
    valid, warnings, report = validate(qpos)
    metadata["validation"] = {"structurally_valid": valid, "warnings": warnings, "report": report}
    if not valid:
        print(report, file=sys.stderr)
        return 1
    print(f"wrote {write_npz(args.output, qpos, metadata)}")
    print(f"qpos {qpos.shape} @ {FPS:g} fps; gestures={[m.gesture_id for m in matches] or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ---------------------------------------------------------------------------
# Speech-synchronised composition
#
# The original `compose` estimates when each word is spoken from a
# words-per-minute figure and then packs the gesture clips back to back. Both
# halves of that are wrong once real audio is involved: the estimate drifts from
# what the voice actually does, and packing the clips together means the motion
# ends while the sentence is still being spoken. The result is a robot gesturing
# at the wrong words and then standing still through the rest of the paragraph.
#
# Edge-TTS will report the exact time of every word if asked for
# `boundary="WordBoundary"` (it defaults to sentence-level, which is why this was
# not obvious). With those timings each gesture can be placed in the window where
# its own trigger phrase is spoken, and held afterwards until the next one is
# due -- so a gesture always finishes before the next begins.
# ---------------------------------------------------------------------------


def rate_for_wpm(words_per_minute: float, baseline_wpm: float = 150.0) -> str:
    """Edge TTS `rate` string ("-20%", "+0%", "+15%") for a words-per-minute target.

    Edge's neural voices speak at roughly 150 wpm by default; the rate is a
    relative adjustment, so the UI's absolute pace is mapped onto it here.
    """
    pct = int(round((float(words_per_minute) / baseline_wpm - 1.0) * 100.0))
    pct = max(-50, min(50, pct))
    return f"{pct:+d}%"


def word_timings(text: str, voice: str, out_mp3: str,
                 rate: str = "+0%") -> tuple[list[tuple[float, float]], float]:
    """Synthesize speech and report when each word is spoken.

    Returns one (start, end) pair per word, in order, plus the audio duration.
    """
    import asyncio

    import edge_tts

    words: list[tuple[float, float]] = []

    async def _run() -> None:
        comm = edge_tts.Communicate(text, voice, rate=rate, boundary="WordBoundary")
        with open(out_mp3, "wb") as handle:
            async for chunk in comm.stream():
                if chunk["type"] == "audio":
                    handle.write(chunk["data"])
                elif chunk["type"] == "WordBoundary":
                    start = chunk["offset"] / 1e7
                    words.append((start, start + chunk["duration"] / 1e7))

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_run())
    finally:
        loop.close()

    duration = words[-1][1] if words else 0.0
    return words, duration


MAX_SEQUENCE_SECONDS = 180.0


def _span_time(word_times: list[tuple[float, float]], n_words: int,
               speech_duration_s: float, w_start: int, w_end: int) -> tuple[float, float]:
    """Speech window for a word range, with a proportional fallback."""
    if word_times and len(word_times) == n_words:
        a = word_times[max(0, min(w_start, len(word_times) - 1))][0]
        b = word_times[max(0, min(w_end - 1, len(word_times) - 1))][1]
        return a, max(b, a + 0.25)
    frac_a = w_start / max(n_words, 1)
    frac_b = w_end / max(n_words, 1)
    return frac_a * speech_duration_s, max(frac_b * speech_duration_s,
                                           frac_a * speech_duration_s + 0.25)


def insert_speech_pauses(src_mp3: str, cuts: list[tuple[float, float]], out_mp3: str) -> float:
    """Rewrite `src_mp3` with silence inserted at the given points.

    `cuts` is a list of (time_in_original_audio, pause_seconds). Pauses are
    placed between words, so the result sounds like a speaker pausing rather
    than a stretched recording -- pitch and speaking rate are untouched, which
    is why this is done instead of time-scaling either the audio or the motion.

    Returns the duration of the written file.
    """
    import av

    with av.open(src_mp3) as container:
        stream = container.streams.audio[0]
        rate = int(stream.rate)
        layout = stream.layout.name
        blocks = [frame.to_ndarray() for frame in container.decode(audio=0)]

    if not blocks:
        raise RuntimeError("no audio decoded from the synthesized speech")

    samples = np.concatenate(blocks, axis=1) if blocks[0].ndim == 2 else \
        np.concatenate(blocks)[None, :]
    channels = samples.shape[0]
    dtype = samples.dtype

    pieces, cursor = [], 0
    for at_s, pause_s in sorted(cuts, key=lambda c: c[0]):
        idx = min(max(int(round(at_s * rate)), cursor), samples.shape[1])
        pieces.append(samples[:, cursor:idx])
        pieces.append(np.zeros((channels, int(round(pause_s * rate))), dtype=dtype))
        cursor = idx
    pieces.append(samples[:, cursor:])
    joined = np.concatenate(pieces, axis=1)

    with av.open(out_mp3, "w") as out:
        ostream = out.add_stream("mp3", rate=rate)
        ostream.layout = layout
        frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(joined),
                                           format=str(stream.format.name), layout=layout)
        frame.rate = rate
        for packet in ostream.encode(frame):
            out.mux(packet)
        for packet in ostream.encode(None):
            out.mux(packet)

    return joined.shape[1] / float(rate)


_CLAUSE_PUNCT = re.compile(r"[.,;:!?\u2014\u2013]|\s-\s")


def clause_boundaries(sentence: str) -> set[int]:
    """Word indices after which the sentence has a clause break.

    Index k in the result means punctuation sits between word k and word k+1,
    so a pause inserted at the end of word k sounds like the speaker pausing at
    a comma or full stop rather than stalling mid-phrase.
    """
    spans = [m.span() for m in re.finditer(r"\w+", sentence, flags=re.UNICODE)]
    out: set[int] = set()
    for k in range(len(spans) - 1):
        between = sentence[spans[k][1]:spans[k + 1][0]]
        if _CLAUSE_PUNCT.search(between):
            out.add(k)
    return out


def _phrases_for(intent: str) -> list[str]:
    """Sentence text that should trigger a gesture."""
    base = intent.strip().lower()
    variants = {base}
    extra = {
        "thank you": {"thanks", "thank you very much", "thank you"},
        "goodbye": {"good bye", "bye", "goodbye"},
        "okay": {"ok", "okay"},
        "yes": {"yeah", "yes"},
        "hello": {"hi", "hey", "hello", "hello everyone", "hi everyone"},
        "please": {"please"},
        "sorry": {"i am sorry", "i'm sorry", "sorry"},
        "welcome": {"you are welcome", "welcome to", "welcome everyone", "welcome"},
        "congratulations": {"congrats", "congratulations"},
        "explain": {"explain", "let me explain", "explanation"},
        "study": {"study", "studying"},
        "learn": {"learn", "learning"},
        "work": {"work", "working", "our project", "project"},
        "robot": {"robot", "robotics", "robotics lab"},
        "be careful": {"be careful", "careful", "stay safe"},
        "help": {"help", "help me", "call help"},
        "look": {"look", "look at"},
        "show": {"show", "show me", "demonstrate"},
    }
    variants |= extra.get(base, set())
    return sorted(variants, key=len, reverse=True)


def compose_natural(sentence: str, catalog: list[dict], matches: list[Match],
                    word_times: list[tuple[float, float]], speech_duration_s: float,
                    *, blend_seconds: float = 0.30,
                    tail_seconds: float = 0.8,
                    gap_seconds: float = 0.20,
                    lag_tolerance_s: float = 0.3,
                    max_total_seconds: float = MAX_SEQUENCE_SECONDS,
                    ) -> tuple[np.ndarray, dict]:
    """Play every matched gesture naturally connected with smooth talking postures."""
    by_id = {item["gesture_id"]: item for item in catalog}
    idle = _idle_pose(catalog)
    n_words = _word_count(sentence)
    blend = max(1, int(round(blend_seconds * FPS)))
    breaks = clause_boundaries(sentence)

    # 1. Place each gesture at its trigger word, at full length, and record the
    #    pause the speech must take whenever the next trigger arrives too soon.
    shift = 0.0
    cuts: list[tuple[float, float]] = []
    placed: list[dict] = []
    skipped: list[dict] = []
    earliest = 0.0          # first frame time the next clip may start (after blend)

    for i, match in enumerate(matches):
        item = by_id[match.gesture_id]
        clip = np.loadtxt(item["source_csv"], delimiter=",").astype(np.float64)
        if clip.ndim == 1:
            clip = clip[None, :]
        if clip.shape[1] != 36:
            raise ValueError(f"{match.gesture_id} has {clip.shape[1]} qpos columns; expected 36")

        t0_orig = _span_time(word_times, n_words, speech_duration_s,
                             match.word_start, match.word_end)[0]
        # Within the tolerance a gesture simply waits for the previous one;
        # beyond it the speech was already paused (below) so this is a no-op.
        word_s = t0_orig + shift        # when the trigger word is now spoken
        t0_new = max(word_s, earliest)
        end_new = t0_new + len(clip) / FPS

        # A sequence that would run past the cap stops here rather than being
        # silently compressed -- the caller is told what was left out.
        if max(end_new, speech_duration_s + shift) + tail_seconds > max_total_seconds:
            skipped.append({"gesture_id": match.gesture_id, "intent": match.intent,
                            "matched_phrase": match.phrase,
                            "reason": f"sequence would exceed the {max_total_seconds:.0f}s limit"})
            continue

        earliest = end_new + gap_seconds + blend_seconds
        if i + 1 < len(matches):
            nxt = matches[i + 1]
            next_orig = _span_time(word_times, n_words, speech_duration_s,
                                   nxt.word_start, nxt.word_end)[0]
            
            # Check if there is clause punctuation between current and next trigger
            between = [k for k in breaks if match.word_end - 1 <= k < nxt.word_start]
            
            if earliest > next_orig + shift + lag_tolerance_s or between:
                deficit = max(earliest - (next_orig + shift + lag_tolerance_s), 0.8 if between else 0.0)
                # Prefer the last clause break between the two triggers: the
                # pause then sits on a comma or full stop and everything after
                # it (the trigger word included) moves by the same amount.
                if between:
                    k = max(between)
                    cut_at = _span_time(word_times, n_words, speech_duration_s, k, k + 1)[1]
                    cut_at = min(cut_at, next_orig)
                else:
                    cut_at = next_orig
                cuts.append((cut_at, deficit))
                shift += deficit

        placed.append({"match": match, "item": item, "clip": clip,
                       "start_s": t0_new, "word_s": word_s})

    speech_end_new = speech_duration_s + shift
    motion_end = max((p["start_s"] + len(p["clip"]) / FPS for p in placed), default=0.0)
    total = int(round((max(speech_end_new, motion_end) + tail_seconds) * FPS))

    # Construct continuous motion timeline
    timeline = np.repeat(idle[None, :], total, axis=0)
    events: list[dict] = []
    talking_base = _talking_pose_base(idle)
    current = talking_base.copy()
    filled_to = 0

    if not placed:
        # No gestures matched: fill entire sequence with stomach-level talking idle + micro-movements
        timeline = generate_talking_idle(talking_base, total, catalog, global_frame_offset=0)
        filled_to = total

    for entry in placed:
        clip, match, item = entry["clip"], entry["match"], entry["item"]
        # Intercept trailing return phase of clip to transition to stomach talking pose instead of side hips
        clip = intercept_clip_to_talking_pose(clip, talking_base)

        f_start = min(int(round(entry["start_s"] * FPS)), total - 1)
        b0 = max(filled_to, f_start - blend)

        # In gaps between gestures during speech: fill with natural Talking Idle micro-movements!
        if b0 > filled_to:
            gap_len = b0 - filled_to
            timeline[filled_to:b0] = generate_talking_idle(current, gap_len, catalog, global_frame_offset=filled_to)
            current = timeline[b0 - 1].copy()
        elif filled_to == 0 and f_start > 0:
            timeline[0:f_start] = generate_talking_idle(current, f_start, catalog, global_frame_offset=0)
            current = timeline[f_start - 1].copy()

        # Blend directly into the next gesture clip
        if b0 < f_start:
            timeline[b0:f_start] = blend_qpos(current, clip[0], f_start - b0)

        end = min(f_start + len(clip), total)
        timeline[f_start:end] = clip[:end - f_start]
        current = timeline[end - 1].copy()
        filled_to = end

        events.append({
            "gesture_id": match.gesture_id,
            "intent": match.intent,
            "matched_phrase": match.phrase,
            "word_span": [match.word_start, match.word_end],
            "speech_start_s": round(entry["word_s"], 3),
            "motion_start_s": round(f_start / FPS, 3),
            "lag_s": round(f_start / FPS - entry["word_s"], 3),
            "motion_end_s": round(end / FPS, 3),
            "played_s": round((end - f_start) / FPS, 3),
            "natural_s": round(len(clip) / FPS, 3),
            "time_scale": round((end - f_start) / len(clip), 3),
            "source_csv": os.path.relpath(item["source_csv"], ROOT),
            "source_validation_warnings": item["validation_warnings"],
        })

    # After gestures finish, maintain stomach-level talking posture with organic micro-movements throughout the rest of speech and tail
    if filled_to < total:
        rem_frames = total - filled_to
        timeline[filled_to:total] = generate_talking_idle(current, rem_frames, catalog, global_frame_offset=filled_to)

    return timeline, {
        "format": "kimodo-g1-sentence-sequence-v3-natural",
        "sentence": sentence,
        "fps": FPS,
        "frames": int(timeline.shape[0]),
        "duration_s": round(total / FPS, 3),
        "speech_offset_s": 0.0,
        "scheduled_speech_end_s": round(speech_end_new, 3),
        "speech_cuts": [[round(a, 3), round(b, 3)] for a, b in cuts],
        "added_pause_s": round(shift, 3),
        "word_timing_source": ("edge-tts WordBoundary"
                               if word_times and len(word_times) == n_words
                               else "proportional fallback"),
        "gestures": events,
        "skipped_gestures": skipped,
    }


def compose_synced(sentence: str, catalog: list[dict], matches: list[Match],
                   word_times: list[tuple[float, float]], speech_duration_s: float,
                   *, blend_seconds: float = 0.25,
                   tail_seconds: float = 0.6,
                   min_gesture_seconds: float = 0.7) -> tuple[np.ndarray, dict]:
    """Lay gestures onto a timeline that runs for the whole spoken paragraph.

    Each gesture is time-scaled into the window where its trigger phrase is
    actually spoken, blended into from whatever pose preceded it, and held until
    the next gesture is due. Between and after gestures the robot holds rather
    than snapping to rest, and it returns to the idle pose only at the very end.
    """
    by_id = {item["gesture_id"]: item for item in catalog}
    idle = _idle_pose(catalog)
    n_words = _word_count(sentence)

    def span_time(w_start: int, w_end: int) -> tuple[float, float]:
        """Actual speech window for a word range, with a proportional fallback."""
        if word_times and len(word_times) == n_words:
            a = word_times[max(0, min(w_start, len(word_times) - 1))][0]
            b = word_times[max(0, min(w_end - 1, len(word_times) - 1))][1]
            return a, max(b, a + 0.25)
        frac_a = w_start / max(n_words, 1)
        frac_b = w_end / max(n_words, 1)
        return frac_a * speech_duration_s, max(frac_b * speech_duration_s,
                                               frac_a * speech_duration_s + 0.25)

    total_s = max(speech_duration_s, 0.5) + tail_seconds
    total = int(round(total_s * FPS))
    blend = max(1, int(round(blend_seconds * FPS)))

    timeline = np.repeat(idle[None, :], total, axis=0)
    events: list[dict] = []
    talking_base = _talking_pose_base(idle)
    current = talking_base.copy()
    filled_to = 0                      # frames already written

    plan = []
    for match in matches:
        a, b = span_time(match.word_start, match.word_end)
        plan.append([match, a, b])

    # Keep gestures strictly sequential: a gesture may run past its phrase into
    # the following silence, but never into the next gesture's window.
    kept: list = []
    skipped: list[dict] = []
    for i, entry in enumerate(plan):
        match, t0, _t1 = entry
        next_start = plan[i + 1][1] if i + 1 < len(plan) else total_s
        room = next_start - blend_seconds - t0
        if room < min_gesture_seconds:
            skipped.append({"gesture_id": match.gesture_id, "intent": match.intent,
                            "matched_phrase": match.phrase,
                            "reason": f"only {max(room, 0):.2f}s before the next "
                                      f"gesture; needs {min_gesture_seconds:.2f}s"})
            continue
        kept.append(entry)

    # Re-cap against the gestures that actually survived.
    for i, entry in enumerate(kept):
        next_start = kept[i + 1][1] if i + 1 < len(kept) else total_s
        entry[2] = min(max(entry[2], entry[1] + min_gesture_seconds),
                       next_start - blend_seconds)
    plan = kept

    if not plan:
        timeline = generate_talking_idle(talking_base, total, catalog, global_frame_offset=0)
        filled_to = total

    for match, t0, t1 in plan:
        item = by_id[match.gesture_id]
        clip = np.loadtxt(item["source_csv"], delimiter=",")
        if clip.ndim == 1:
            clip = clip[None, :]
        if clip.shape[1] != 36:
            raise ValueError(f"{match.gesture_id} has {clip.shape[1]} qpos columns; expected 36")

        f_start = int(round(t0 * FPS))
        f_end = min(int(round(t1 * FPS)), total)
        n = max(6, f_end - f_start)
        clip = resample_clip(clip.astype(np.float64), n)
        clip = intercept_clip_to_talking_pose(clip, talking_base)

        # Blend in over the frames just before the word is spoken
        b0 = max(filled_to, f_start - blend)
        if b0 > filled_to:
            gap_len = b0 - filled_to
            timeline[filled_to:b0] = generate_talking_idle(current, gap_len, catalog, global_frame_offset=filled_to)
            current = timeline[b0 - 1].copy()
        elif filled_to == 0 and f_start > 0:
            timeline[0:f_start] = generate_talking_idle(current, f_start, catalog, global_frame_offset=0)
            current = timeline[f_start - 1].copy()

        if b0 < f_start:
            timeline[b0:f_start] = blend_qpos(current, clip[0], f_start - b0)

        end = min(f_start + n, total)
        timeline[f_start:end] = clip[:end - f_start]
        current = clip[min(n, end - f_start) - 1]
        filled_to = end

        events.append({
            "gesture_id": match.gesture_id,
            "intent": match.intent,
            "matched_phrase": match.phrase,
            "word_span": [match.word_start, match.word_end],
            "speech_start_s": round(t0, 3),
            "motion_start_s": round(f_start / FPS, 3),
            "motion_end_s": round(end / FPS, 3),
            "source_csv": os.path.relpath(item["source_csv"], ROOT),
            "source_validation_warnings": item["validation_warnings"],
        })

    # Hold stomach talking pose with micro-movements for any remaining speech/tail
    if filled_to < total:
        rem_frames = total - filled_to
        timeline[filled_to:total] = generate_talking_idle(current, rem_frames, catalog, global_frame_offset=filled_to)

    return timeline, {
        "format": "kimodo-g1-sentence-sequence-v2-synced",
        "sentence": sentence,
        "fps": FPS,
        "frames": int(timeline.shape[0]),
        "duration_s": round(total / FPS, 3),
        "speech_offset_s": 0.0,
        "scheduled_speech_end_s": round(speech_duration_s, 3),
        "word_timing_source": ("edge-tts WordBoundary"
                               if word_times and len(word_times) == n_words
                               else "proportional fallback"),
        "gestures": events,
        "skipped_gestures": skipped,
    }
