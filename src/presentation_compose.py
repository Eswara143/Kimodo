#!/usr/bin/env python3
"""Presentation mode: speak a paragraph and gesture with allowed library clips.

Pipeline (no Kimodo model, no text encoder -- retrieval and composition only):

    text -> clause/word tokens -> phrase matching against the allowlist
         -> Edge TTS audio + word timestamps (onset-calibrated on the audio)
         -> schedule: main action of each gesture on its spoken phrase,
            pauses inserted at phrase boundaries when a gesture needs time
         -> motion: attention -> gestures / talking idle -> attention
         -> continuity + stationary-stance validation -> NPZ/CSV/WAV/MP4

Presentation text is speech content, never a command: candidates are only
ever drawn from ``presentation_safety.build_allowlist()`` and anything else
becomes a diagnostic.  See presentation_safety.py for what "allowed" means.

    python presentation_compose.py "Hello! Welcome to robotics." --out outputs/presentations/demo

Simulation-side only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
import wave
from dataclasses import dataclass, field, asdict

import numpy as np

import presentation_safety as ps
from presentation_safety import (ARMS, FPS, LEGS, ROOT_POS, ROOT_QUAT, WAIST,
                                 RESTRICTED_INTENT_TO_CLASS, RESTRICTED_VOCABULARY)

ROOT = ps.ROOT
OUT_ROOT = os.path.join(ROOT, "outputs", "presentations")
MAX_TOTAL_SECONDS = 180.0

# --------------------------------------------------------------------------
# Tunables (seconds unless stated).  All of them are reported in schedule.json.
# --------------------------------------------------------------------------
DEFAULTS = {
    "attention_hold_s": 0.6,      # verified attention pose held before anything moves
    "attention_return_s": 0.9,    # blend back to attention after the last phrase
    "attention_tail_s": 0.7,      # hold attention at the very end
    "phrase_pause_s": 0.8,        # user-configurable pause at sentence boundaries (. ! ?)
    "min_entry_s": 0.30,          # shortest preparation into a main action
    "min_exit_s": 0.35,           # shortest recovery out of a main action
    "min_transition_s": 0.45,     # shortest direct gesture-to-gesture transition
    "lag_tolerance_s": 0.25,      # main action may start this late after its word before we pause the speech
    "lead_tolerance_s": 0.10,     # ...or this early
    "talk_blend_s": 0.6,          # attention -> talking pose
    "min_talk_s": 0.4,            # shortest talking-idle stretch worth inserting between gestures
    "repeat_cooldown_s": 8.0,     # same gesture is not re-triggered within this window
    "speech_tail_s": 0.3,         # talking continues this long after the last word
}

# Relaxed talking pose on top of the attention pose (arms only), chosen on the
# loaded model with forward kinematics.  On g1skel34 the elbow joint reads 0 at
# ~90 deg of flexion and grows as the forearm drops, so raising the forearm
# means *lowering* the elbow value; the sign is the same on both arms (only
# roll/yaw mirror).  elbow 0.10 -> -0.10 and shoulder_pitch 0.20 -> 0.30 lifts
# each forearm an additional +14 deg (from -28 deg to -14 deg below horizontal),
# hands ~5 cm higher comfortably in front of the stomach / lower chest, elbows
# relaxed beside the torso, no self-contact.
TALKING_ARMS = {
    "shoulder_pitch": 0.30, "shoulder_roll": 0.12, "shoulder_yaw": -0.35,
    "elbow": -0.10, "wrist_roll": 0.0, "wrist_pitch": 0.0, "wrist_yaw": 0.0,
}
# Slow conversational motion around that pose: (amplitude rad, frequency Hz)
# per arm joint.  The amplitudes are the v1 baseline; IDLE_AMPLITUDE_SCALE
# widens the range (not the speed -- frequencies are untouched).
# Boosted to 1.56 (25% increase over 1.25 baseline) for enhanced visibility.
IDLE_MOTION = {
    "shoulder_pitch": (0.045, 0.21), "shoulder_roll": (0.025, 0.17), "shoulder_yaw": (0.035, 0.23),
    "elbow": (0.09, 0.27), "wrist_roll": (0.14, 0.31), "wrist_pitch": (0.09, 0.25), "wrist_yaw": (0.07, 0.19),
}
IDLE_AMPLITUDE_SCALE = 1.56
ARM_JOINT_ORDER = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"]
# Mirror sign for the right arm (roll / yaw axes are mirrored on the G1).
RIGHT_SIGN = {"shoulder_pitch": 1, "shoulder_roll": -1, "shoulder_yaw": -1, "elbow": 1,
              "wrist_roll": -1, "wrist_pitch": 1, "wrist_yaw": -1}

# --------------------------------------------------------------------------
# Trigger lexicon.  Only gesture ids that pass the allowlist are ever used;
# an entry whose clip is denied turns into a diagnostic, not a gesture.
# "strong" phrases match anywhere; "weak" ones only at the start of a clause,
# because words like "right", "up" or "now" are mostly not gestures mid-sentence.
# --------------------------------------------------------------------------
LEXICON: list[tuple[str, str, str]] = [
    # (gesture_id, strength, phrase)
    ("PDF_001", "strong", "hello"), ("PDF_001", "strong", "hello everyone"), ("PDF_001", "strong", "hello everybody"),
    ("PDF_001", "strong", "greetings"), ("PDF_002", "strong", "hi"), ("PDF_002", "strong", "hi everyone"),
    ("PDF_003", "strong", "hey"), ("PDF_004", "strong", "good morning"), ("PDF_005", "strong", "good afternoon"),
    ("PDF_006", "strong", "good evening"), ("PDF_007", "strong", "welcome"), ("PDF_007", "strong", "welcome to"),
    ("PDF_007", "strong", "welcome everyone"), ("PDF_007", "strong", "you are all welcome"),
    ("PDF_010", "strong", "bye"), ("PDF_010", "strong", "goodbye"), ("PDF_010", "strong", "good bye"),
    ("PDF_010", "strong", "farewell"), ("PDF_011", "strong", "see you"), ("PDF_011", "strong", "see you later"),
    ("PDF_011", "strong", "see you soon"), ("PDF_012", "strong", "come again"), ("PDF_012", "strong", "visit again"),
    ("PDF_014", "strong", "thank you"), ("PDF_014", "strong", "thanks"), ("PDF_014", "strong", "thank you very much"),
    ("PDF_014", "strong", "thank you all"), ("PDF_014", "strong", "many thanks"), ("PDF_014", "strong", "grateful"),
    ("PDF_016", "strong", "please"), ("PDF_016", "strong", "kindly"),
    ("PDF_021", "weak", "yes"), ("PDF_021", "weak", "yeah"), ("PDF_021", "strong", "absolutely"), ("PDF_021", "strong", "of course"),
    ("PDF_023", "weak", "okay"), ("PDF_023", "weak", "ok"), ("PDF_023", "weak", "alright"), ("PDF_023", "weak", "all right"),
    ("PDF_025", "strong", "wrong"), ("PDF_025", "strong", "incorrect"), ("PDF_025", "strong", "mistake"),
    ("PDF_026", "strong", "maybe"), ("PDF_026", "strong", "perhaps"), ("PDF_026", "strong", "possibly"),
    ("PDF_027", "strong", "i don't know"), ("PDF_027", "strong", "i do not know"), ("PDF_027", "strong", "no idea"),
    ("PDF_027", "strong", "not sure"), ("PDF_028", "strong", "i know"), ("PDF_028", "strong", "as you know"),
    ("PDF_028", "strong", "we know"), ("PDF_031", "strong", "i don't understand"), ("PDF_031", "strong", "i do not understand"),
    ("PDF_033", "weak", "what"), ("PDF_035", "weak", "where"), ("PDF_036", "weak", "who"), ("PDF_039", "weak", "which"),
    ("PDF_041", "strong", "repeat"), ("PDF_041", "strong", "once again"), ("PDF_041", "weak", "again"),
    ("PDF_042", "strong", "speak"), ("PDF_042", "strong", "speaking"), ("PDF_043", "strong", "listen"),
    ("PDF_043", "strong", "listen carefully"), ("PDF_044", "strong", "quiet"), ("PDF_044", "strong", "silence"),
    ("PDF_046", "strong", "on the left"), ("PDF_046", "strong", "to the left"), ("PDF_046", "strong", "left side"),
    ("PDF_047", "strong", "on the right"), ("PDF_047", "strong", "to the right"), ("PDF_047", "strong", "right side"),
    ("PDF_048", "strong", "straight ahead"), ("PDF_048", "strong", "ahead of us"), ("PDF_050", "strong", "up there"),
    ("PDF_050", "strong", "above us"), ("PDF_052", "strong", "right here"), ("PDF_052", "strong", "over here"),
    ("PDF_054", "strong", "nearby"), ("PDF_054", "strong", "close by"), ("PDF_056", "weak", "straight"),
    ("PDF_068", "strong", "inside"), ("PDF_072", "strong", "stop here"), ("PDF_080", "strong", "finish"),
    ("PDF_080", "strong", "finished"), ("PDF_080", "strong", "the end"), ("PDF_080", "strong", "complete"),
    ("PDF_082", "strong", "pause"), ("PDF_083", "strong", "slowly"), ("PDF_083", "strong", "slow down"),
    ("PDF_084", "strong", "quickly"), ("PDF_084", "strong", "fast"), ("PDF_084", "strong", "quick"),
    ("PDF_086", "weak", "stay"), ("PDF_094", "strong", "watch"), ("PDF_094", "strong", "observe"),
    ("PDF_095", "strong", "show me"), ("PDF_218", "strong", "show you"), ("PDF_218", "strong", "let me show"),
    ("PDF_218", "strong", "show"), ("PDF_218", "strong", "showing"), ("PDF_096", "weak", "give"), ("PDF_097", "weak", "take"),
    ("PDF_101", "weak", "drop"), ("PDF_107", "weak", "lower"),
    ("PDF_111", "strong", "happy"), ("PDF_111", "strong", "glad"), ("PDF_111", "strong", "pleased"), ("PDF_111", "strong", "delighted"),
    ("PDF_112", "strong", "sad"), ("PDF_112", "strong", "unfortunately"), ("PDF_113", "strong", "excited"),
    ("PDF_113", "strong", "exciting"), ("PDF_113", "strong", "thrilled"), ("PDF_116", "strong", "confused"),
    ("PDF_116", "strong", "confusing"), ("PDF_119", "strong", "brave"), ("PDF_119", "strong", "courage"),
    ("PDF_122", "strong", "tired"), ("PDF_122", "strong", "exhausted"), ("PDF_124", "strong", "bored"),
    ("PDF_124", "strong", "boring"), ("PDF_125", "strong", "interested"), ("PDF_125", "strong", "interesting"),
    ("PDF_130", "strong", "calm"), ("PDF_130", "strong", "relax"), ("PDF_130", "strong", "relaxed"),
    ("PDF_134", "strong", "amazing"), ("PDF_134", "strong", "awesome"), ("PDF_134", "strong", "incredible"),
    ("PDF_134", "strong", "wonderful"), ("PDF_135", "strong", "great"), ("PDF_135", "strong", "excellent"),
    ("PDF_135", "strong", "fantastic"), ("PDF_135", "strong", "brilliant"), ("PDF_138", "strong", "funny"),
    ("PDF_138", "strong", "hilarious"), ("PDF_141", "strong", "painful"), ("PDF_141", "strong", "it hurts"),
    ("PDF_142", "strong", "strong"), ("PDF_142", "strong", "powerful"), ("PDF_142", "strong", "strength"),
    ("PDF_143", "strong", "weak"), ("PDF_144", "strong", "victory"), ("PDF_144", "strong", "we won"),
    ("PDF_144", "strong", "success"), ("PDF_144", "strong", "successful"), ("PDF_146", "strong", "myself"),
    ("PDF_146", "strong", "my name is"), ("PDF_146", "strong", "i am"), ("PDF_146", "strong", "i'm"),
    ("PDF_148", "strong", "all of us"), ("PDF_148", "strong", "our team"), ("PDF_148", "strong", "together"),
    ("PDF_152", "strong", "separate"), ("PDF_152", "strong", "separately"), ("PDF_152", "strong", "apart"),
    ("PDF_154", "strong", "help me"), ("PDF_155", "strong", "i will help"), ("PDF_155", "strong", "we can help"),
    ("PDF_155", "strong", "happy to help"), ("PDF_155", "strong", "help you"), ("PDF_157", "strong", "give me"),
    ("PDF_170", "strong", "trust me"), ("PDF_170", "strong", "believe me"), ("PDF_171", "strong", "eat"),
    ("PDF_171", "strong", "eating"), ("PDF_171", "strong", "food"), ("PDF_172", "strong", "drink"),
    ("PDF_172", "strong", "drinking"), ("PDF_176", "strong", "clean"), ("PDF_176", "strong", "cleaning"),
    ("PDF_177", "strong", "brush"), ("PDF_180", "strong", "mix"), ("PDF_180", "strong", "mixing"),
    ("PDF_181", "strong", "write"), ("PDF_181", "strong", "writing"), ("PDF_182", "strong", "read"),
    ("PDF_182", "strong", "reading"), ("PDF_183", "strong", "book"), ("PDF_184", "strong", "phone"),
    ("PDF_184", "strong", "call me"), ("PDF_188", "weak", "work"), ("PDF_188", "strong", "hard work"),
    ("PDF_199", "strong", "search"), ("PDF_199", "strong", "searching"), ("PDF_199", "strong", "look for"),
    ("PDF_201", "strong", "take a photo"), ("PDF_201", "strong", "photo"), ("PDF_201", "strong", "picture"),
    ("PDF_201", "strong", "camera"), ("PDF_203", "strong", "open the door"), ("PDF_205", "strong", "knock"),
    ("PDF_207", "strong", "study"), ("PDF_207", "strong", "studying"), ("PDF_208", "strong", "teach"),
    ("PDF_208", "strong", "teaching"), ("PDF_208", "strong", "lesson"), ("PDF_209", "strong", "explain"),
    ("PDF_209", "strong", "let me explain"), ("PDF_209", "strong", "explanation"), ("PDF_209", "strong", "describe"),
    ("PDF_209", "strong", "in detail"), ("PDF_210", "strong", "question"), ("PDF_210", "strong", "questions"),
    ("PDF_210", "strong", "any questions"), ("PDF_212", "strong", "think"), ("PDF_212", "strong", "thinking"),
    ("PDF_212", "strong", "consider"), ("PDF_213", "strong", "idea"), ("PDF_213", "strong", "ideas"),
    ("PDF_213", "strong", "concept"), ("PDF_214", "strong", "remember"), ("PDF_214", "strong", "recall"),
    ("PDF_215", "strong", "forget"), ("PDF_215", "strong", "forgot"), ("PDF_216", "strong", "read this"),
    ("PDF_217", "strong", "write this"), ("PDF_219", "strong", "demonstrate"), ("PDF_219", "strong", "demonstration"),
    ("PDF_219", "strong", "demo"), ("PDF_220", "strong", "present"), ("PDF_220", "strong", "presentation"),
    ("PDF_220", "strong", "presenting"), ("PDF_220", "strong", "introduce"), ("PDF_220", "strong", "introducing"),
    ("PDF_221", "strong", "ask"), ("PDF_221", "strong", "asking"), ("PDF_222", "strong", "tell me"),
    ("PDF_222", "strong", "tell you"), ("PDF_222", "strong", "let me tell you"), ("PDF_223", "strong", "good job"),
    ("PDF_223", "strong", "well done"), ("PDF_223", "strong", "great job"), ("PDF_223", "strong", "nice work"),
    ("PDF_225", "strong", "done"), ("PDF_228", "strong", "emergency"), ("PDF_231", "strong", "don't move"),
    ("PDF_231", "strong", "do not move"), ("PDF_231", "strong", "hold still"), ("PDF_232", "strong", "careful"),
    ("PDF_232", "strong", "carefully"), ("PDF_232", "strong", "caution"), ("PDF_233", "strong", "be careful"),
    ("PDF_233", "strong", "watch out"), ("PDF_235", "strong", "unsafe"), ("PDF_235", "strong", "dangerous"),
    ("PDF_235", "strong", "danger"), ("PDF_236", "strong", "fire"), ("PDF_237", "strong", "call for help"),
    ("PDF_237", "strong", "call help"), ("PDF_240", "weak", "stop"), ("PDF_240", "strong", "stop immediately"),
    ("PDF_240", "strong", "stop now"), ("PDF_241", "weak", "now"), ("PDF_241", "strong", "right now"),
    ("PDF_242", "strong", "later"), ("PDF_242", "strong", "afterwards"), ("PDF_245", "weak", "first"),
    ("PDF_245", "strong", "first of all"), ("PDF_245", "strong", "firstly"), ("PDF_247", "weak", "last"),
    ("PDF_247", "strong", "lastly"), ("PDF_247", "strong", "finally"),
]

# Phrases whose library clip is denied by the allowlist: they never gesture,
# but naming them here gives the user a precise "blocked" diagnostic.
LEXICON += [
    ("PDF_017", "strong", "sorry"), ("PDF_017", "strong", "i am sorry"), ("PDF_017", "strong", "apologies"),
    ("PDF_019", "strong", "congratulations"), ("PDF_019", "strong", "congrats"), ("PDF_008", "strong", "namaste"),
    ("PDF_013", "strong", "nice to meet you"), ("PDF_018", "strong", "excuse me"), ("PDF_022", "weak", "no"),
    ("PDF_024", "strong", "correct"), ("PDF_029", "strong", "understand"), ("PDF_159", "strong", "handshake"),
    ("PDF_159", "strong", "shake hands"), ("PDF_160", "strong", "high five"), ("PDF_161", "strong", "hug"),
    ("PDF_164", "strong", "clap"), ("PDF_165", "strong", "applause"), ("PDF_165", "strong", "round of applause"),
    ("PDF_145", "strong", "celebrate"), ("PDF_163", "strong", "bow"), ("PDF_009", "strong", "wave goodbye"),
    ("PDF_226", "strong", "danger ahead"), ("PDF_229", "strong", "help!"),
]

NEGATION_CUES = {"not", "no", "never", "don't", "dont", "doesn't", "doesnt", "didn't", "didnt", "won't", "wont",
                 "can't", "cant", "cannot", "isn't", "isnt", "aren't", "arent", "wasn't", "wasnt", "without",
                 "neither", "nor", "shouldn't", "shouldnt", "mustn't", "mustnt", "couldn't", "couldnt"}
# Text between these is reported speech / a quotation, not the speaker's own gesture.
QUOTE_PAIRS = [('"', '"'), ("“", "”"), ("'", "'"), ("‘", "’")]

_WORD_RE = re.compile(r"[A-Za-z0-9À-ɏЀ-ӿఀ-౿]+(?:['’][A-Za-z]+)?", re.UNICODE)
_SENTENCE_END = re.compile(r"[.!?…]|\n")
_CLAUSE_BREAK = re.compile(r"[.!?,;:—–…()\n]|\s-\s")


# --------------------------------------------------------------------------
# Text analysis
# --------------------------------------------------------------------------
@dataclass
class Word:
    idx: int
    text: str
    start: int
    end: int
    clause: int = 0
    sentence: int = 0
    clause_start: bool = False
    sentence_end: bool = False
    clause_end: bool = False
    quoted: bool = False


def tokenize(text: str) -> list[Word]:
    words = [Word(i, m.group(0), m.start(), m.end()) for i, m in enumerate(_WORD_RE.finditer(text))]
    clause = sentence = 0
    depth = {q: False for q, _ in QUOTE_PAIRS}
    prev_end = 0
    for w in words:
        between = text[prev_end:w.start]
        if w.idx == 0 or _CLAUSE_BREAK.search(between):
            w.clause_start = True
        if w.idx > 0 and _CLAUSE_BREAK.search(between):
            clause += 1
        if w.idx > 0 and _SENTENCE_END.search(between):
            sentence += 1
        for ch in between:
            for open_q, close_q in QUOTE_PAIRS:
                if ch == open_q and open_q == close_q:
                    depth[open_q] = not depth[open_q]
                elif ch == open_q:
                    depth[open_q] = True
                elif ch == close_q:
                    depth[open_q] = False
        w.clause, w.sentence = clause, sentence
        w.quoted = any(depth.values())
        prev_end = w.end
    for i, w in enumerate(words):
        nxt = words[i + 1] if i + 1 < len(words) else None
        after = text[w.end: nxt.start] if nxt else text[w.end:]
        w.sentence_end = bool(_SENTENCE_END.search(after)) or nxt is None
        w.clause_end = bool(_CLAUSE_BREAK.search(after)) or nxt is None
    return words


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower().replace("’", "'"))


@dataclass
class Candidate:
    phrase: str
    word_start: int
    word_end: int
    gesture_id: str | None
    intent: str | None
    strength: str
    status: str = "candidate"     # scheduled | negated | quoted | blocked_restricted | blocked_allowlist | repeat_suppressed | timing_conflict | invalid_transition | asset_missing | crowded
    category: str = ""
    reason: str = ""
    corrective: str = ""
    detail: dict = field(default_factory=dict)


def find_candidates(text: str, words: list[Word], allowlist: dict) -> list[Candidate]:
    """Every phrase that could be a gesture, each with a status explaining why it will or will not play."""
    clips = allowlist["clips"]
    lower = text.lower().replace("’", "'")
    char_to_word = {}
    for w in words:
        for c in range(w.start, w.end):
            char_to_word[c] = w.idx
    by_word_start = {w.start: w.idx for w in words}

    raw: list[Candidate] = []
    # 1. Lexicon phrases -> allowed gestures (and allowlist-denied clips of the library).
    library_intents = {re.sub(r"[^\w\s]", "", c["intent"]).strip().lower(): gid for gid, c in clips.items() if c.get("intent")}
    seen_spans: set[tuple[int, int, str]] = set()
    entries = list(LEXICON)
    # Denied library intents that are not in the lexicon still deserve a
    # diagnostic when they are spoken as a cue (clause start, or a multi-word
    # phrase); allowed gestures are only ever triggered through the lexicon.
    lexicon_phrases = {p for _, _, p in LEXICON}
    for intent, gid in library_intents.items():
        if intent in lexicon_phrases:
            continue
        if not clips[gid].get("allowed"):
            entries.append((gid, "strong" if " " in intent else "auto", intent))
        elif (clips[gid].get("number") or 0) >= 251:
            # The 251-1500 batch names descriptive mimes ("play a guitar",
            # "piston stroke", "rising trend", "give an example"): the intent itself is the cue.
            entries.append((gid, "strong", intent))
            if intent == "piston stroke":
                entries.append((gid, "strong", "pistol stroke"))

    for gid, strength, phrase in entries:
        pat = re.compile(r"(?<![a-z0-9])" + re.escape(phrase).replace(r"\ ", r"\s+") + r"(?![a-z0-9])")
        for m in pat.finditer(lower):
            ws, we = char_to_word.get(m.start()), char_to_word.get(m.end() - 1)
            if ws is None or we is None:
                continue
            key = (ws, we + 1, gid)
            if key in seen_spans:
                continue
            seen_spans.add(key)
            clip = clips.get(gid, {})
            raw.append(Candidate(text[words[ws].start:words[we].end], ws, we + 1, gid, clip.get("intent"), strength))
    # 2. Restricted vocabulary -> diagnostics only.
    for phrase in sorted(RESTRICTED_VOCABULARY, key=len, reverse=True):
        pat = re.compile(r"(?<![a-z0-9])" + re.escape(phrase).replace(r"\ ", r"\s+") + r"(?![a-z0-9])")
        for m in pat.finditer(lower):
            ws, we = char_to_word.get(m.start()), char_to_word.get(m.end() - 1)
            if ws is None or we is None:
                continue
            key = (ws, we + 1, "RESTRICTED")
            if key in seen_spans:
                continue
            seen_spans.add(key)
            raw.append(Candidate(text[words[ws].start:words[we].end], ws, we + 1, None, None, "restricted",
                                 status="blocked_restricted", category="blocked_restricted",
                                 reason=f"'{phrase}' names a whole-body action; presentation mode never performs "
                                        "walking, jumping, kicking, flips, turns or floor motions",
                                 corrective="No action needed -- the phrase is spoken only. Use the explicit "
                                            "action-command controls if you really want that motion executed."))

    # Longest phrase wins for overlapping spans; among equals a restricted-action
    # word outranks a library clip so the diagnostic names the real reason.
    raw.sort(key=lambda c: (c.word_start, -(c.word_end - c.word_start), 0 if c.status == "blocked_restricted" else 1))
    out: list[Candidate] = []
    occupied: list[tuple[int, int]] = []
    for c in raw:
        if any(c.word_start < e and c.word_end > s for s, e in occupied):
            continue
        # Auto-added single-word denied intents ("you", "there", "go"...) are
        # everyday words; they only matter as a cue at the start of a clause.
        if c.strength == "auto" and not words[c.word_start].clause_start:
            continue
        occupied.append((c.word_start, c.word_end))
        out.append(c)

    for c in out:
        if c.status == "blocked_restricted":
            gid = library_intents.get(c.phrase.lower())
            if gid:
                c.detail = {"library_clip": gid, "allowlist_status": clips[gid]["status"],
                            "allowlist_reasons": clips[gid].get("reasons", [])}
            continue
        clip = clips.get(c.gesture_id, {})
        w0 = words[c.word_start]
        if not clip:
            c.status, c.category = "asset_missing", "asset_missing"
            c.reason = f"gesture {c.gesture_id} is not in the library manifest"
            c.corrective = "Regenerate the library entry or remove it from the lexicon."
        elif clip.get("status", "").startswith("asset_"):
            c.status, c.category = "asset_missing", "asset_missing"
            c.reason = f"{c.gesture_id} ({clip.get('intent')}): {'; '.join(clip.get('reasons', []))}"
            c.corrective = "Regenerate this gesture (generate_all_250_gestures.py) so its CSV exists and validates."
        elif not clip.get("allowed") and c.strength == "weak" and not w0.clause_start:
            c.status, c.category = "weak_context", "no_gesture"
            c.reason = f"'{c.phrase}' mid-clause is not treated as a gesture cue"
            c.corrective = "No action needed."
        elif not clip.get("allowed"):
            cls = clip.get("restricted_class")
            c.status = "blocked_restricted" if cls else "blocked_allowlist"
            c.category = c.status
            c.reason = (f"{c.gesture_id} ({clip.get('intent')}) is excluded from presentation mode: "
                        + "; ".join(clip.get("reasons", [])))
            c.corrective = ("Spoken only. A separately generated and validated stationary version of this "
                            "gesture would be needed to use it in a presentation.")
        elif w0.quoted:
            c.status, c.category = "quoted", "no_gesture"
            c.reason = "phrase is inside quotation marks (reported speech), so it is spoken without a gesture"
            c.corrective = "Remove the quotation marks if the robot should gesture on it."
        elif _negated(words, c):
            c.status, c.category = "negated", "no_gesture"
            c.reason = f"'{c.phrase}' is negated in its clause, so the gesture would contradict the words"
            c.corrective = "No action needed."
        elif c.strength == "weak" and not w0.clause_start:
            c.status, c.category = "weak_context", "no_gesture"
            c.reason = f"'{c.phrase}' only counts as a gesture cue at the start of a clause"
            c.corrective = "Put it at the start of a clause (e.g. 'Now, ...') if it should gesture."
        else:
            c.status, c.category = "scheduled", "gesture"
    # Prefer the more specific cue: a single generic word ("show") right before
    # a multi-word phrase ("rising trend") in the same clause would otherwise
    # take the slot and leave the specific gesture with no room.
    sched = [c for c in out if c.status == "scheduled"]
    for a, b in zip(sched, sched[1:]):
        if (words[a.word_start].clause == words[b.word_start].clause and b.word_start - a.word_end < 4
                and a.word_end - a.word_start == 1 and b.word_end - b.word_start >= 2):
            a.status, a.category = "crowded", "no_gesture"
            a.reason = f"'{a.phrase}' yields to the more specific cue '{b.phrase}' right after it"
            a.corrective = "No action needed."
    return out


def _negated(words: list[Word], c: Candidate) -> bool:
    w0 = words[c.word_start]
    k = c.word_start - 1
    steps = 0
    while k >= 0 and words[k].clause == w0.clause and steps < 3:
        if _norm(words[k].text) in {_norm(x) for x in NEGATION_CUES}:
            return True
        k -= 1
        steps += 1
    return False


# --------------------------------------------------------------------------
# Speech: Edge TTS with word boundaries, then onset calibration on the audio
# --------------------------------------------------------------------------
def synthesize(text: str, voice: str, rate: str, out_mp3: str) -> tuple[list[dict], float]:
    import asyncio
    import edge_tts

    events: list[dict] = []

    async def _run() -> None:
        comm = edge_tts.Communicate(text, voice, rate=rate, boundary="WordBoundary")
        with open(out_mp3, "wb") as handle:
            async for chunk in comm.stream():
                if chunk["type"] == "audio":
                    handle.write(chunk["data"])
                elif chunk["type"] == "WordBoundary":
                    start = chunk["offset"] / 1e7
                    events.append({"text": chunk["text"], "start": start, "end": start + chunk["duration"] / 1e7})

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(_run())
    finally:
        loop.close()
    return events, decode_audio(out_mp3)[1]


def decode_audio(path: str) -> tuple[np.ndarray, float, int]:
    """Mono float32 PCM, duration, sample rate."""
    import av
    with av.open(path) as container:
        stream = container.streams.audio[0]
        sr = int(stream.rate)
        blocks = [fr.to_ndarray() for fr in container.decode(audio=0)]
    if not blocks:
        raise RuntimeError(f"no audio decoded from {path}")
    x = np.concatenate(blocks, axis=1) if blocks[0].ndim == 2 else np.concatenate(blocks)[None, :]
    x = x.astype(np.float32)
    if np.abs(x).max() > 1.5:
        x = x / 32768.0
    mono = x.mean(axis=0)
    return mono, len(mono) / sr, sr


def align_words(words: list[Word], events: list[dict]) -> list[tuple[float, float] | None]:
    """Map Edge's word events onto our tokens; None where a token had no event."""
    import difflib
    a = [_norm(w.text) for w in words]
    b = [_norm(e["text"]) for e in events]
    out: list[tuple[float, float] | None] = [None] * len(words)
    if a == b:
        return [(e["start"], e["end"]) for e in events]
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                out[i1 + k] = (events[j1 + k]["start"], events[j1 + k]["end"])
        elif tag == "replace" and (i2 - i1) == (j2 - j1):
            for k in range(i2 - i1):
                out[i1 + k] = (events[j1 + k]["start"], events[j1 + k]["end"])
        elif tag == "replace" and j2 > j1:
            # several tokens for one event (or vice versa): spread the event span across them
            t0, t1 = events[j1]["start"], events[j2 - 1]["end"]
            n = i2 - i1
            for k in range(n):
                out[i1 + k] = (t0 + (t1 - t0) * k / n, t0 + (t1 - t0) * (k + 1) / n)
    # Interpolate anything still missing between neighbours.
    for i in range(len(out)):
        if out[i] is None:
            prev = next((out[j] for j in range(i - 1, -1, -1) if out[j]), None)
            nxt = next((out[j] for j in range(i + 1, len(out)) if out[j]), None)
            if prev and nxt:
                out[i] = (prev[1], nxt[0])
            elif prev:
                out[i] = (prev[1], prev[1] + 0.25)
            elif nxt:
                out[i] = (max(nxt[0] - 0.25, 0.0), nxt[0])
    return out


def calibrate_timings(pcm: np.ndarray, sr: int, timings: list[tuple[float, float]]) -> tuple[list[tuple[float, float]], dict]:
    """Snap word starts to the audio.

    Edge's WordBoundary offsets run ahead of the audible onset by a roughly
    constant amount (~130 ms measured here).  For every word that follows a
    silence the true onset is detected from the envelope; the median lead is
    applied to all other words.  Returns the corrected timings and a report.
    """
    win = max(1, int(0.005 * sr))
    env = np.convolve(np.abs(pcm), np.ones(win) / win, mode="same")
    thr = float(env.max()) * 0.03 if env.size else 0.0
    leads, snapped = [], {}
    for i, (t0, t1) in enumerate(timings):
        i0 = int(t0 * sr)
        pre = env[max(0, i0 - int(0.12 * sr)): max(1, i0)]
        if i0 >= len(env) or (pre.size and pre.max() > thr):
            continue                 # no silence before it: nothing to detect
        seg = env[i0: min(len(env), i0 + int(0.35 * sr))]
        j = int(np.argmax(seg > thr)) if (seg > thr).any() else None
        if j is None:
            continue
        lead = j / sr
        if 0.0 <= lead <= 0.3:
            leads.append(lead)
            snapped[i] = t0 + lead
    shift = float(np.median(leads)) if leads else 0.0
    shift = min(max(shift, 0.0), 0.25)
    fixed = []
    for i, (t0, t1) in enumerate(timings):
        s = snapped.get(i, t0 + shift)
        fixed.append((s, max(t1 + shift, s + 0.05)))
    for i in range(len(fixed) - 1):        # keep words ordered and non-overlapping
        if fixed[i][1] > fixed[i + 1][0]:
            fixed[i] = (fixed[i][0], fixed[i + 1][0])
    return fixed, {"onset_shift_s": round(shift, 3), "onsets_detected": len(leads),
                   "detected_leads_ms": [round(l * 1000) for l in leads[:12]]}


def rate_for_wpm(wpm: float, baseline: float = 150.0) -> str:
    pct = int(round((float(wpm) / baseline - 1.0) * 100.0))
    return f"{max(-50, min(50, pct)):+d}%"


# --------------------------------------------------------------------------
# Motion helpers
# --------------------------------------------------------------------------
def smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def qnorm(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q)


def slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    a, b = qnorm(a), qnorm(b)
    d = float(np.dot(a, b))
    if d < 0:
        b, d = -b, -d
    if d > 0.9995:
        return qnorm((1 - t) * a + t * b)
    th = math.acos(min(1.0, d))
    return (math.sin((1 - t) * th) * a + math.sin(t * th) * b) / math.sin(th)


def hermite(p0: np.ndarray, v0: np.ndarray, p1: np.ndarray, v1: np.ndarray, frames: int) -> np.ndarray:
    """C1 transition of `frames` frames from (p0, v0) to (p1, v1); frame `frames` would equal p1.

    Velocities are per second.  The root quaternion is slerped with a smoothstep.
    """
    n = max(int(frames), 1)
    T = n / FPS
    out = np.empty((n, len(p0)), dtype=np.float64)
    for i in range(n):
        t = i / n
        h00, h10 = 2 * t**3 - 3 * t**2 + 1, t**3 - 2 * t**2 + t
        h01, h11 = -2 * t**3 + 3 * t**2, t**3 - t**2
        out[i] = h00 * p0 + h10 * T * v0 + h01 * p1 + h11 * T * v1
        out[i, ROOT_QUAT] = slerp(p0[ROOT_QUAT], p1[ROOT_QUAT], float(smoothstep(np.array(t))))
    return out


def velocity_at(clip: np.ndarray, idx: int) -> np.ndarray:
    idx = max(0, min(idx, len(clip) - 1))
    if len(clip) < 2:
        return np.zeros_like(clip[0])
    if idx == 0:
        v = (clip[1] - clip[0]) * FPS
    elif idx == len(clip) - 1:
        v = (clip[-1] - clip[-2]) * FPS
    else:
        v = (clip[idx + 1] - clip[idx - 1]) * FPS / 2
    v = v.copy()
    v[ROOT_QUAT] = 0.0
    return v


def arm_vector(spec: dict, right: bool) -> np.ndarray:
    return np.array([spec[j] * (RIGHT_SIGN[j] if right else 1) for j in ARM_JOINT_ORDER])


class IdleTrack:
    """Relaxed talking pose with slow, smooth conversational motion and still moments.

    pose(f) is a deterministic function of the frame index (seeded), so any
    segment sampled from it is continuous with any other and the exits/entries
    around it can target the exact pose and velocity at their boundary frame.
    This is a *fallback* motion, not a semantic gesture.
    """

    def __init__(self, attention: np.ndarray, seed: int = 7):
        self.base = attention.copy()
        self.base[22:29] = arm_vector(TALKING_ARMS, right=False)
        self.base[29:36] = arm_vector(TALKING_ARMS, right=True)
        rng = np.random.default_rng(seed)
        self.phase = rng.uniform(0, 2 * math.pi, size=(2, len(ARM_JOINT_ORDER)))
        self.freq_jitter = rng.uniform(0.85, 1.15, size=(2, len(ARM_JOINT_ORDER)))
        # Gate schedule: moving stretches with still moments in between.
        t, gates = 0.0, []
        while t < MAX_TOTAL_SECONDS + 10:
            move, still = rng.uniform(2.4, 4.2), rng.uniform(0.7, 1.5)
            gates.append((t, t + move, t + move + still))
            t += move + still
        self.gates = gates

    def gate(self, t: float) -> float:
        for a, b, c in self.gates:
            if a <= t < c:
                if t < b:
                    ramp = 0.6
                    up = smoothstep(np.array((t - a) / ramp))
                    down = smoothstep(np.array((b - t) / ramp))
                    return float(min(up, down))
                return 0.0
        return 0.0

    def pose(self, f: int) -> np.ndarray:
        t = f / FPS
        p = self.base.copy()
        g = self.gate(t)
        for side, sl in ((0, slice(22, 29)), (1, slice(29, 36))):
            off = np.array([
                IDLE_AMPLITUDE_SCALE * IDLE_MOTION[j][0]
                * math.sin(2 * math.pi * IDLE_MOTION[j][1] * self.freq_jitter[side][k] * t + self.phase[side][k])
                for k, j in enumerate(ARM_JOINT_ORDER)])
            if side == 1:
                off = off * np.array([RIGHT_SIGN[j] for j in ARM_JOINT_ORDER])
            p[sl] = p[sl] + g * off
        return p

    def vel(self, f: int) -> np.ndarray:
        return (self.pose(f + 1) - self.pose(max(f - 1, 0))) * FPS / (2 if f > 0 else 1)

    def block(self, f0: int, f1: int) -> np.ndarray:
        return np.stack([self.pose(f) for f in range(f0, f1)]) if f1 > f0 else np.zeros((0, 36))


def attention_pose(allowlist: dict) -> tuple[np.ndarray, dict]:
    """The verified standing pose: first frame of an allowed greeting clip, re-centred at the origin."""
    for gid in ("PDF_001", "PDF_209", "PDF_007"):
        clip = allowlist["clips"].get(gid)
        if clip and clip.get("allowed"):
            q = np.loadtxt(os.path.join(ROOT, clip["csv"]), delimiter=",")[0].astype(np.float64)
            q[0:2] = 0.0
            q[ROOT_QUAT] = qnorm(q[ROOT_QUAT])
            m = ps.measure(q[None, :])
            reasons = ps.judge(m, ps.joint_limit_violations(q[None, :]))
            if not reasons:
                return q, {"source": f"{gid} frame 0", "measurements": m}
    raise RuntimeError("no allowed clip available to derive the attention pose from")


# --------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------
@dataclass
class Placed:
    cand: Candidate
    clip: np.ndarray
    phases: dict
    main_start_f: int = 0
    main_end_f: int = 0
    word_start_s: float = 0.0
    word_end_s: float = 0.0
    lag_s: float = 0.0
    entry_f: int = 0
    exit_f: int = 0
    pre_transitioned: bool = False


class Cuts:
    """Silences inserted into the speech, in original-audio time."""

    def __init__(self):
        self.items: list[list] = []   # [t_orig, dur, label]

    def add(self, t: float, dur: float, label: str) -> None:
        for it in self.items:
            if abs(it[0] - t) < 1e-6:
                it[1] += dur
                it[2] = f"{it[2]}+{label}" if label not in it[2] else it[2]
                return
        self.items.append([t, dur, label])
        self.items.sort(key=lambda it: it[0])

    def final(self, t: float, inclusive: bool = True) -> float:
        """Original-audio time -> final time.  A pause placed exactly at a word's
        end belongs after that word: use inclusive=False for end times."""
        if inclusive:
            return t + sum(d for at, d, _ in self.items if at <= t + 1e-9)
        return t + sum(d for at, d, _ in self.items if at < t - 1e-9)

    def total(self) -> float:
        return sum(d for _, d, _ in self.items)


def plan_and_build(text: str, words: list[Word], cands: list[Candidate], timings: list[tuple[float, float]],
                   speech_len: float, allowlist: dict, cfg: dict, excluded: set[int] | None = None) -> dict:
    """One planning + building pass.  Raises _Replan(cand_index) when a transition fails."""
    excluded = excluded or set()
    clips = allowlist["clips"]
    attention, att_info = attention_pose(allowlist)
    idle = IdleTrack(attention)
    cuts = Cuts()
    lead_in = cfg["attention_hold_s"]
    cuts.add(0.0, lead_in, "lead-in")
    if cfg["phrase_pause_s"] > 0:
        for w in words[:-1]:
            if w.sentence_end and w.idx < len(timings):
                cuts.add(timings[w.idx][1], cfg["phrase_pause_s"], "sentence pause")

    def f_of(t: float) -> int:
        return int(round(t * FPS))

    placed: list[Placed] = []
    last_use: dict[str, float] = {}
    prev: Placed | None = None
    for ci, c in enumerate(cands):
        if c.status != "scheduled" or ci in excluded:
            if ci in excluded and c.status == "scheduled":
                pass
            continue
        clip_meta = clips[c.gesture_id]
        clip = np.loadtxt(os.path.join(ROOT, clip_meta["csv"]), delimiter=",").astype(np.float64)
        # The clip was admitted on its full-body measurement (feet planted,
        # no drift).  In the composed timeline its waist and arms are played on
        # the verified attention stance: root and legs stay exactly where the
        # presentation started, so every clip shares one stance and no
        # transition ever has to slide a foot between two slightly different
        # Kimodo stances.
        clip[:, ROOT_POS] = attention[ROOT_POS]
        clip[:, ROOT_QUAT] = attention[ROOT_QUAT]
        clip[:, LEGS] = attention[LEGS]
        ph = clip_meta["phases"]
        m0, m1 = ph["main"]
        main_len = (m1 - m0) / FPS
        prep_len, rec_len = m0 / FPS, (len(clip) - m1) / FPS
        entry_s = max(prep_len, cfg["min_entry_s"])
        exit_s = max(rec_len, cfg["min_exit_s"])

        ws_orig, we_orig = timings[c.word_start][0], timings[c.word_end - 1][1]
        # Repetition: the same gesture again within the cooldown is spoken only.
        if c.gesture_id in last_use and ws_orig - last_use[c.gesture_id] < cfg["repeat_cooldown_s"]:
            c.status, c.category = "repeat_suppressed", "no_gesture"
            c.reason = f"{c.intent} was already performed {ws_orig - last_use[c.gesture_id]:.1f}s earlier"
            c.corrective = "No action needed; repeated cues within a few seconds play once."
            continue

        ws = cuts.final(ws_orig)
        if prev is None:
            earliest = lead_in + entry_s
        else:
            earliest = prev.main_end_f / FPS + cfg["min_transition_s"]
        phrase_len = max(cuts.final(we_orig, False) - ws, 0.15)
        allowed_lag = min(cfg["lag_tolerance_s"], phrase_len)
        deficit = earliest - ws
        if deficit > allowed_lag:
            # The speech has to wait.  Put the pause on the last phrase boundary
            # between the previous trigger and this one so it reads as a breath.
            lo = prev.cand.word_end - 2 if prev else -1
            boundary = None
            for k in range(c.word_start - 1, lo, -1):
                if words[k].clause_end and k < len(timings):
                    boundary = k
                    break
            if prev is None:
                cuts.add(0.0, deficit, "lead-in")
            elif boundary is not None:
                cuts.add(timings[boundary][1], deficit, f"pause for '{c.phrase}'")
            else:
                c.status, c.category = "timing_conflict", "timing_conflict"
                c.reason = (f"'{c.phrase}' needs {deficit:.2f}s more after '{prev.cand.phrase}' and there is no "
                            "comma or full stop between them to pause the speech at")
                c.corrective = ("Add punctuation between the two phrases, lower the speech rate, or remove one "
                                "of the two cues. Gestures are never cut short or sped up to fit.")
                continue
            ws = cuts.final(ws_orig)
            deficit = earliest - ws
        main_start = max(ws, earliest)
        main_end = main_start + main_len
        if main_end + exit_s + cfg["attention_return_s"] + cfg["attention_tail_s"] > MAX_TOTAL_SECONDS:
            c.status, c.category = "timing_conflict", "timing_conflict"
            c.reason = f"the presentation would exceed the {MAX_TOTAL_SECONDS:.0f}s limit at '{c.phrase}'"
            c.corrective = "Shorten the text or split it into two presentations."
            continue
        p = Placed(c, clip, ph, f_of(main_start), f_of(main_start) + (m1 - m0), ws, cuts.final(we_orig, False),
                   round(main_start - ws, 3), f_of(entry_s), f_of(exit_s))
        placed.append(p)
        last_use[c.gesture_id] = ws_orig
        prev = p

    speech_end = cuts.final(speech_len) if timings else lead_in
    # ------------------------------------------------------------------ build
    end_talk = speech_end + cfg["speech_tail_s"]
    last_motion = max([p.main_end_f / FPS + p.exit_f / FPS for p in placed] + [end_talk])
    total_f = f_of(last_motion + cfg["attention_return_s"] + cfg["attention_tail_s"]) + 1
    timeline = np.repeat(attention[None, :], total_f, axis=0)
    segments: list[dict] = []
    transitions: list[dict] = []

    def seg(kind: str, a: int, b: int, **extra) -> None:
        if b > a:
            segments.append({"kind": kind, "start_f": int(a), "end_f": int(b), "start_s": round(a / FPS, 3),
                             "end_s": round(b / FPS, 3), **extra})

    def write(a: int, block: np.ndarray) -> None:
        b = min(a + len(block), total_f)
        if b > a:
            timeline[a:b] = block[: b - a]

    def check_collision(block: np.ndarray, label: str) -> str | None:
        m = ps.measure(block)
        reasons = ps.judge(m, ps.joint_limit_violations(block))
        return "; ".join(reasons) if reasons else None

    att_f = f_of(cfg["attention_hold_s"])
    seg("attention", 0, att_f, label="attention hold")
    cur_f = att_f
    cur_pose, cur_vel = attention.copy(), np.zeros(36)
    in_talking = False

    for gi, p in enumerate(placed):
        gap_f = p.main_start_f - cur_f
        entry_pose = p.clip[p.phases["main"][0]]
        entry_vel = velocity_at(p.clip, p.phases["main"][0])
        route_via_idle = (not p.pre_transitioned) and gap_f >= p.entry_f + f_of(cfg["talk_blend_s"] if not in_talking else 0) + f_of(cfg["min_talk_s"])
        if route_via_idle:
            talk_start = cur_f
            if not in_talking:
                bl = f_of(cfg["talk_blend_s"])
                block = hermite(cur_pose, cur_vel, idle.pose(cur_f + bl), idle.vel(cur_f + bl), bl)
                write(cur_f, block)
                seg("transition", cur_f, cur_f + bl, label="attention -> talking")
                transitions.append({"kind": "to_talking", "start_f": cur_f, "end_f": cur_f + bl, "label": "attention -> talking"})
                talk_start = cur_f + bl
            entry_start = p.main_start_f - p.entry_f
            write(talk_start, idle.block(talk_start, entry_start))
            seg("talking", talk_start, entry_start, label="talking idle (fallback)")
            cur_f, cur_pose, cur_vel = entry_start, idle.pose(entry_start), idle.vel(entry_start)
            in_talking = True
        if not p.pre_transitioned:
            # Entry: from the current pose into the main action.  From the
            # attention pose the clip's own preparation is used (it starts
            # there); from the talking pose a C1 Hermite path goes straight to
            # the main-action start so the hands never drop to the sides first.
            n_entry = p.main_start_f - cur_f
            if n_entry < f_of(cfg["min_entry_s"]) - 1:
                raise _Replan(cands.index(p.cand), f"only {n_entry / FPS:.2f}s available to prepare '{p.cand.phrase}'")
            prep = p.clip[: p.phases["main"][0]]
            from_attention = not in_talking and np.allclose(cur_pose[ARMS], attention[ARMS], atol=1e-6)
            if from_attention and len(prep) >= f_of(0.25) and abs(len(prep) - n_entry) <= max(2, int(0.5 * len(prep))):
                prep_rt = resample(prep, n_entry)
                s = smoothstep(np.arange(n_entry) / max(n_entry - 1, 1))[:, None]
                block = prep_rt + (cur_pose - prep_rt[0]) * (1 - s)
                for i in range(n_entry):
                    block[i, ROOT_QUAT] = slerp(cur_pose[ROOT_QUAT], prep_rt[i, ROOT_QUAT], float(s[i, 0]))
                kind = "clip preparation" if n_entry == len(prep) else f"clip preparation retimed x{n_entry / len(prep):.2f}"
            else:
                block = hermite(cur_pose, cur_vel, entry_pose, entry_vel, n_entry)
                kind = "hermite"
            bad = check_collision(np.concatenate([block, entry_pose[None, :]]), "entry")
            if bad:
                # Try again through the relaxed pose as a clearance waypoint.
                half = n_entry // 2
                wp = idle.base
                block = np.concatenate([hermite(cur_pose, cur_vel, wp, np.zeros(36), half),
                                        hermite(wp, np.zeros(36), entry_pose, entry_vel, n_entry - half)])
                kind = "hermite via talking pose"
                bad = check_collision(np.concatenate([block, entry_pose[None, :]]), "entry")
                if bad:
                    raise _Replan(cands.index(p.cand), f"no collision-free path into '{p.cand.phrase}': {bad}")
            write(cur_f, block)
            seg("entry", cur_f, p.main_start_f, gesture_id=p.cand.gesture_id, intent=p.cand.intent, label=f"prepare {p.cand.intent} ({kind})")
            transitions.append({"kind": "entry", "start_f": cur_f, "end_f": p.main_start_f, "label": f"into {p.cand.intent}"})
        # Main action, untouched.
        m0, m1 = p.phases["main"]
        write(p.main_start_f, p.clip[m0:m1])
        hold = p.phases.get("hold")
        seg("gesture", p.main_start_f, p.main_end_f, gesture_id=p.cand.gesture_id, intent=p.cand.intent,
            phrase=p.cand.phrase, label=f"{p.cand.intent}: main action",
            hold_s=[round((hold[0] - m0) / FPS + p.main_start_f / FPS, 3), round((hold[1] - m0) / FPS + p.main_start_f / FPS, 3)] if hold else None)
        cur_f, cur_pose, cur_vel = p.main_end_f, p.clip[m1 - 1], velocity_at(p.clip, m1 - 1)
        in_talking = False

        nxt = placed[gi + 1] if gi + 1 < len(placed) else None
        if nxt is not None:
            gap_f = nxt.main_start_f - cur_f
            if gap_f >= p.exit_f + nxt.entry_f + f_of(cfg["min_talk_s"]):
                # Enough room: recover into the talking pose, idle, then the next entry.
                # The clip's own recovery drops the hands to the sides; it is
                # replaced by a C1 path from the last main-action frame straight
                # into the talking pose (where the idle track is at that frame).
                n_exit = p.exit_f
                tgt_f = cur_f + n_exit
                tgt_pose, tgt_vel = idle.pose(tgt_f), idle.vel(tgt_f)
                block = hermite(cur_pose, cur_vel, tgt_pose, tgt_vel, n_exit)
                kind = "hermite"
                bad = check_collision(np.concatenate([block, tgt_pose[None, :]]), "exit")
                if bad:
                    half = n_exit // 2
                    block = np.concatenate([hermite(cur_pose, cur_vel, idle.base, np.zeros(36), half),
                                            hermite(idle.base, np.zeros(36), tgt_pose, tgt_vel, n_exit - half)])
                    kind = "hermite via talking pose"
                    bad = check_collision(np.concatenate([block, tgt_pose[None, :]]), "exit")
                    if bad:
                        raise _Replan(cands.index(p.cand), f"no collision-free recovery out of '{p.cand.phrase}': {bad}")
                write(cur_f, block)
                seg("exit", cur_f, tgt_f, gesture_id=p.cand.gesture_id, intent=p.cand.intent, label=f"recover {p.cand.intent} -> talking ({kind})")
                transitions.append({"kind": "exit", "start_f": cur_f, "end_f": tgt_f, "label": f"out of {p.cand.intent}"})
                cur_f, cur_pose, cur_vel, in_talking = tgt_f, tgt_pose, tgt_vel, True
                # talking idle until the next entry is written by the next iteration (route_via_idle)
            else:
                # Direct transition: keep this main action whole, drop its neutral
                # return and the next one's neutral start, and move between them.
                n_tr = gap_f
                if n_tr < f_of(cfg["min_transition_s"]) - 1:
                    raise _Replan(cands.index(nxt.cand), f"only {n_tr / FPS:.2f}s between '{p.cand.phrase}' and '{nxt.cand.phrase}'")
                nm0 = nxt.phases["main"][0]
                tgt_pose, tgt_vel = nxt.clip[nm0], velocity_at(nxt.clip, nm0)
                block = hermite(cur_pose, cur_vel, tgt_pose, tgt_vel, n_tr)
                kind = "direct hermite"
                bad = check_collision(np.concatenate([block, tgt_pose[None, :]]), "transition")
                if bad:
                    half = n_tr // 2
                    block = np.concatenate([hermite(cur_pose, cur_vel, idle.base, np.zeros(36), half),
                                            hermite(idle.base, np.zeros(36), tgt_pose, tgt_vel, n_tr - half)])
                    kind = "hermite via talking pose"
                    bad = check_collision(np.concatenate([block, tgt_pose[None, :]]), "transition")
                    if bad:
                        raise _Replan(cands.index(nxt.cand), f"no collision-free transition into '{nxt.cand.phrase}': {bad}")
                write(cur_f, block)
                seg("transition", cur_f, nxt.main_start_f, gesture_id=nxt.cand.gesture_id, intent=nxt.cand.intent,
                    label=f"{p.cand.intent} -> {nxt.cand.intent} ({kind})")
                transitions.append({"kind": "direct", "start_f": cur_f, "end_f": nxt.main_start_f, "label": f"{p.cand.intent} -> {nxt.cand.intent}"})
                cur_f, cur_pose, cur_vel = nxt.main_start_f, tgt_pose, tgt_vel
                nxt.pre_transitioned = True

    # ------------------------------------------------------------- ending
    end_talk_f = f_of(end_talk)
    if placed and cur_f + placed[-1].exit_f < end_talk_f - f_of(cfg["min_talk_s"]) and not in_talking:
        p = placed[-1]
        n_exit = p.exit_f
        tgt_f = cur_f + n_exit
        tgt_pose, tgt_vel = idle.pose(tgt_f), idle.vel(tgt_f)
        block = hermite(cur_pose, cur_vel, tgt_pose, tgt_vel, n_exit)
        if check_collision(block, "exit"):
            raise _Replan(cands.index(p.cand), f"no collision-free recovery out of '{p.cand.phrase}'")
        write(cur_f, block)
        seg("exit", cur_f, tgt_f, gesture_id=p.cand.gesture_id, intent=p.cand.intent, label=f"recover {p.cand.intent} -> talking")
        transitions.append({"kind": "exit", "start_f": cur_f, "end_f": tgt_f, "label": f"out of {p.cand.intent}"})
        cur_f, cur_pose, cur_vel, in_talking = tgt_f, tgt_pose, tgt_vel, True
    if not placed and end_talk_f > cur_f + f_of(cfg["talk_blend_s"]) + f_of(cfg["min_talk_s"]):
        bl = f_of(cfg["talk_blend_s"])
        block = hermite(cur_pose, cur_vel, idle.pose(cur_f + bl), idle.vel(cur_f + bl), bl)
        write(cur_f, block)
        seg("transition", cur_f, cur_f + bl, label="attention -> talking")
        transitions.append({"kind": "to_talking", "start_f": cur_f, "end_f": cur_f + bl, "label": "attention -> talking"})
        cur_f = cur_f + bl
        cur_pose, cur_vel, in_talking = idle.pose(cur_f), idle.vel(cur_f), True
    if in_talking and end_talk_f > cur_f:
        write(cur_f, idle.block(cur_f, end_talk_f))
        seg("talking", cur_f, end_talk_f, label="talking idle (fallback)")
        cur_f, cur_pose, cur_vel = end_talk_f, idle.pose(end_talk_f), idle.vel(end_talk_f)
    # Return to attention and hold.
    n_ret = max(f_of(cfg["attention_return_s"]), placed[-1].exit_f if placed and not in_talking else 0)
    block = hermite(cur_pose, cur_vel, attention, np.zeros(36), n_ret)
    if check_collision(block, "return"):
        # go via the relaxed pose
        half = n_ret // 2
        block = np.concatenate([hermite(cur_pose, cur_vel, idle.base, np.zeros(36), half),
                                hermite(idle.base, np.zeros(36), attention, np.zeros(36), n_ret - half)])
        if check_collision(block, "return"):
            raise _Replan(cands.index(placed[-1].cand) if placed else -1, "no collision-free return to attention")
    write(cur_f, block)
    seg("transition", cur_f, cur_f + n_ret, label="return to attention")
    transitions.append({"kind": "return", "start_f": cur_f, "end_f": cur_f + n_ret, "label": "return to attention"})
    cur_f += n_ret
    total_f = cur_f + f_of(cfg["attention_tail_s"])
    timeline = timeline[:total_f] if total_f <= len(timeline) else np.concatenate([timeline, np.repeat(attention[None, :], total_f - len(timeline), axis=0)])
    timeline[cur_f:] = attention
    seg("attention", cur_f, total_f, label="attention hold")

    return {"timeline": timeline, "segments": segments, "transitions": transitions, "placed": placed,
            "cuts": cuts, "speech_end_s": speech_end, "attention": attention, "attention_info": att_info,
            "idle": idle, "total_f": total_f}


class _Replan(Exception):
    def __init__(self, cand_index: int, reason: str):
        super().__init__(reason)
        self.cand_index, self.reason = cand_index, reason


def resample(block: np.ndarray, n: int) -> np.ndarray:
    n = max(int(n), 1)
    if len(block) == n:
        return block.copy()
    xo = np.linspace(0, 1, len(block))
    xn = np.linspace(0, 1, n)
    out = np.empty((n, block.shape[1]))
    for c in range(block.shape[1]):
        out[:, c] = np.interp(xn, xo, block[:, c])
    for i in range(n):
        out[i, ROOT_QUAT] = qnorm(out[i, ROOT_QUAT])
    return out


# --------------------------------------------------------------------------
# Audio assembly
# --------------------------------------------------------------------------
def build_audio(pcm: np.ndarray, sr: int, cuts: Cuts, total_s: float, out_wav: str) -> float:
    pieces, cursor = [], 0
    for at, dur, _ in cuts.items:
        idx = min(max(int(round(at * sr)), cursor), len(pcm))
        pieces.append(pcm[cursor:idx])
        pieces.append(np.zeros(int(round(dur * sr)), dtype=np.float32))
        cursor = idx
    pieces.append(pcm[cursor:])
    y = np.concatenate(pieces)
    need = int(round(total_s * sr))
    if len(y) < need:
        y = np.concatenate([y, np.zeros(need - len(y), dtype=np.float32)])
    with wave.open(out_wav, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(y, -1, 1) * 32767).astype("<i2").tobytes())
    return len(y) / sr


# --------------------------------------------------------------------------
# Rendering (paced) and muxing
# --------------------------------------------------------------------------
def render_paced(qpos: np.ndarray, out_mp4: str, fps: float = FPS, width: int = 640, height: int = 480,
                 target_fps: float = 120.0, progress=None, azimuth: float = 165.0) -> dict:
    """MuJoCo EGL render of the timeline, paced so the GPU/CPU are not saturated.

    Same camera as render_motion_video.render but fixed (no orbit): a presenter
    is watched from the front (azimuth=165) or side (azimuth=90).
    """
    import av
    import mujoco
    ctx = ps.load_model()
    model, data = ctx["model"], ctx["data"]
    renderer = mujoco.Renderer(model, height=height, width=width)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance, cam.elevation, cam.azimuth = 2.25, -6.0, azimuth
    cam.lookat[:] = [0.0, 0.0, 0.92]
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)
    container = av.open(out_mp4, mode="w")
    stream = container.add_stream("libx264", rate=int(round(fps)))
    stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
    stream.options = {"crf": "21", "profile": "baseline", "preset": "veryfast", "threads": "2"}
    t0 = time.time()
    n = len(qpos)
    for i, fr in enumerate(qpos):
        data.qpos[:] = fr
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam, scene_option=opt)
        img = renderer.render()
        for pkt in stream.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
            container.mux(pkt)
        # pacing: never render faster than target_fps on average
        ahead = (i + 1) / target_fps - (time.time() - t0)
        if ahead > 0.002:
            time.sleep(min(ahead, 0.05))
        if progress and i % 60 == 0:
            progress(i, n)
    for pkt in stream.encode():
        container.mux(pkt)
    container.close()
    return {"render_s": round(time.time() - t0, 2), "frames": int(n), "render_fps": round(n / max(time.time() - t0, 1e-6), 1)}


def mux(video: str, wav: str, out_mp4: str, total_s: float) -> None:
    import imageio_ffmpeg
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", video, "-i", wav, "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
           "-af", f"apad=whole_dur={total_s:.3f}", "-t", f"{total_s:.3f}", "-movflags", "+faststart", out_mp4]
    res = subprocess.run(cmd, capture_output=True, timeout=180)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg mux failed: {res.stderr.decode(errors='replace')[-400:]}")


# --------------------------------------------------------------------------
# Top level
# --------------------------------------------------------------------------
def prepare(text: str, out_dir: str, voice: str = "en-US-AvaNeural", words_per_minute: float = 150.0,
            phrase_pause_s: float | None = None, target_duration_s: float | None = None,
            render: bool = True, progress=None, name: str | None = None) -> dict:
    cfg = dict(DEFAULTS)
    if phrase_pause_s is not None:
        cfg["phrase_pause_s"] = float(phrase_pause_s)
    t_start = time.time()
    os.makedirs(out_dir, exist_ok=True)
    name = name or os.path.basename(out_dir.rstrip("/"))
    diagnostics: list[dict] = []
    stages: dict[str, float] = {}

    def tick(label: str, frac: float) -> None:
        if progress:
            progress(label, frac)

    tick("classifying library", 0.02)
    t = time.time()
    allowlist = ps.build_allowlist()
    stages["allowlist_s"] = round(time.time() - t, 2)

    words = tokenize(text)
    if not words:
        raise ValueError("The text contains no words.")

    if target_duration_s is not None and target_duration_s > 0:
        pause_count = sum(1 for w in words if any(p in w.text for p in ".!?"))
        est_pause_time = pause_count * cfg.get("phrase_pause_s", 0.8)
        avail_speech_time = max(1.0, float(target_duration_s) - est_pause_time - 1.2)
        calc_wpm = (len(words) / avail_speech_time) * 60.0
        words_per_minute = max(80.0, min(220.0, calc_wpm))


    cands = find_candidates(text, words, allowlist)

    # ---- speech
    tick("synthesizing speech", 0.10)
    t = time.time()
    raw_mp3 = os.path.join(out_dir, "tts_raw.mp3")
    speech_ok, timings, calib = True, [], {}
    try:
        events, speech_len = synthesize(text, voice, rate_for_wpm(words_per_minute), raw_mp3)
        pcm, speech_len, sr = decode_audio(raw_mp3)
        timings = align_words(words, events)
        timings = [tm if tm else (0.0, 0.05) for tm in timings]
        timings, calib = calibrate_timings(pcm, sr, timings)
        timing_source = f"edge-tts WordBoundary ({len(events)} events) + audio onset calibration"
    except Exception as exc:  # noqa: BLE001
        speech_ok = False
        diagnostics.append({"category": "speech_failure", "phrase": text[:60], "reason": f"{type(exc).__name__}: {exc}",
                            "corrective": "Edge TTS needs network access. Check the connection and prepare again; "
                                          "the motion below is planned on estimated word timing and has no audio."})
        sr, pcm = 24000, np.zeros(0, dtype=np.float32)
        spw = 60.0 / max(words_per_minute, 1)
        timings = [(0.3 + i * spw, 0.3 + (i + 0.8) * spw) for i in range(len(words))]
        speech_len = 0.3 + len(words) * spw
        timing_source = "proportional estimate (speech synthesis failed)"
    stages["tts_s"] = round(time.time() - t, 2)

    # ---- plan + build, re-planning without a gesture whose transition cannot be made safe
    tick("planning gestures", 0.30)
    t = time.time()
    excluded: set[int] = set()
    for _attempt in range(12):
        for c in cands:
            if c.status in ("timing_conflict", "repeat_suppressed", "invalid_transition") and cands.index(c) not in excluded:
                c.status, c.category, c.reason, c.corrective = "scheduled", "gesture", "", ""
        try:
            built = plan_and_build(text, words, cands, timings, speech_len, allowlist, cfg, excluded)
            break
        except _Replan as rp:
            if rp.cand_index < 0 or rp.cand_index in excluded:
                raise RuntimeError(f"could not build a safe timeline: {rp.reason}")
            excluded.add(rp.cand_index)
            c = cands[rp.cand_index]
            c.status, c.category = "invalid_transition", "invalid_transition"
            c.reason = rp.reason
            c.corrective = "The gesture is spoken only. Add a pause before it or move it to its own sentence."
    else:
        raise RuntimeError("planning did not converge")
    stages["plan_build_s"] = round(time.time() - t, 2)

    timeline, cuts = built["timeline"], built["cuts"]
    total_s = len(timeline) / FPS
    final_words = [{"idx": w.idx, "text": w.text, "start_s": round(cuts.final(timings[w.idx][0]), 3),
                    "end_s": round(cuts.final(timings[w.idx][1], False), 3), "clause": w.clause, "sentence": w.sentence,
                    "quoted": w.quoted} for w in words]

    # ---- validate
    tick("validating motion", 0.55)
    t = time.time()
    check = ps.check_timeline(timeline, built["transitions"])
    stages["validate_s"] = round(time.time() - t, 2)
    # Also the project's structural validator, the tool of record.
    csv_path = os.path.join(out_dir, f"{name}.csv")
    np.savetxt(csv_path, timeline, delimiter=",")
    validator = _run_validator(csv_path)
    ok = check["ok"] and validator["returncode"] == 0

    # ---- diagnostics from candidates
    for c in cands:
        if c.status == "scheduled":
            continue
        diagnostics.append({"category": c.category or c.status, "status": c.status, "phrase": c.phrase,
                            "word_span": [c.word_start, c.word_end], "gesture_id": c.gesture_id, "intent": c.intent,
                            "reason": c.reason, "corrective": c.corrective, **({"detail": c.detail} if c.detail else {})})
    scheduled_words = {i for c in cands if c.status == "scheduled" for i in range(c.word_start, c.word_end)}
    if not built["placed"]:
        diagnostics.append({"category": "no_gesture", "phrase": text[:80], "reason": "no phrase matched an allowed gesture; "
                            "the relaxed talking motion is used for the whole presentation",
                            "corrective": "Optional: include cue words such as hello, welcome, thank you, explain, question."})
    for r in check["reasons"]:
        diagnostics.append({"category": "validation_failure", "phrase": "(whole timeline)", "reason": r,
                            "corrective": "The presentation is blocked. Report this text; the composer should never produce it."})
    if validator["returncode"] != 0:
        diagnostics.append({"category": "validation_failure", "phrase": "(whole timeline)",
                            "reason": "validate_g1_motion.py failed: " + " | ".join(l for l in validator["stdout"].splitlines() if l.startswith("FAIL")),
                            "corrective": "The presentation is blocked."})
    for w in check.get("warnings", []) + [w for w in validator["stdout"].splitlines()
                                          if w.startswith("WARNING") and "peak joint" not in w]:
        diagnostics.append({"category": "review_warning", "phrase": "(whole timeline)", "reason": w,
                            "corrective": "Review threshold only (project placeholder, not a datasheet figure); playback is allowed."})

    # ---- gesture events for the UI
    events_out = []
    for p in built["placed"]:
        m0, m1 = p.phases["main"]
        events_out.append({
            "gesture_id": p.cand.gesture_id, "intent": p.cand.intent, "phrase": p.cand.phrase,
            "word_span": [p.cand.word_start, p.cand.word_end],
            "word_start_s": round(p.word_start_s, 3), "word_end_s": round(p.word_end_s, 3),
            "main_start_s": round(p.main_start_f / FPS, 3), "main_end_s": round(p.main_end_f / FPS, 3),
            "main_lag_s": round(p.main_start_f / FPS - p.word_start_s, 3),
            "apex_s": round((p.main_start_f + p.phases["apex"] - m0) / FPS, 3),
            "phases_in_clip": {k: p.phases[k] for k in ("prep", "main", "hold", "recovery")},
            "source_csv": allowlist["clips"][p.cand.gesture_id]["csv"],
            "review_warnings": allowlist["clips"][p.cand.gesture_id].get("review_warnings", []),
        })

    schedule = {
        "format": "kimodo-g1-presentation-v1",
        "name": name, "text": text, "voice": voice, "words_per_minute": words_per_minute,
        "tts_rate": rate_for_wpm(words_per_minute), "config": cfg, "fps": FPS,
        "frames": int(len(timeline)), "duration_s": round(total_s, 3),
        "speech_start_s": round(cuts.final(timings[0][0]) if timings else 0.0, 3),
        "speech_end_s": round(built["speech_end_s"], 3),
        "word_timing_source": timing_source, "timing_calibration": calib,
        "speech_ok": speech_ok,
        "pauses": [{"at_original_s": round(a, 3), "at_final_s": round(cuts.final(a) - d, 3), "duration_s": round(d, 3), "label": l}
                   for a, d, l in cuts.items],
        "words": final_words,
        "gestures": events_out,
        "segments": built["segments"],
        "diagnostics": diagnostics,
        "validation": {"ok": ok, "stationary_and_continuity": check, "validator": validator},
        "attention_pose": built["attention_info"]["source"],
        "talking_pose_arms": TALKING_ARMS,
        "idle_amplitude_scale": IDLE_AMPLITUDE_SCALE,
        "allowlist_summary": allowlist["summary"],
        "candidates": [asdict(c) for c in cands],
    }

    # ---- artifacts
    npz_path = os.path.join(out_dir, f"{name}.npz")
    np.savez_compressed(npz_path, qpos=timeline.astype(np.float32), fps=np.asarray(FPS, dtype=np.float32),
                        text=np.asarray(text), schedule_json=np.asarray(json.dumps(schedule, ensure_ascii=False)))
    wav_path = os.path.join(out_dir, "speech.wav")
    audio_len = build_audio(pcm, sr, cuts, total_s, wav_path) if speech_ok else None
    schedule["audio_duration_s"] = round(audio_len, 3) if audio_len else None

    if render and ok:
        tick("rendering preview", 0.65)
        silent = os.path.join(out_dir, "preview_silent.mp4")
        rinfo = render_paced(timeline, silent, progress=lambda i, n: tick(f"rendering preview ({i}/{n})", 0.65 + 0.3 * i / max(n, 1)))
        stages.update(rinfo)
        preview = os.path.join(out_dir, "preview.mp4")
        if speech_ok:
            try:
                mux(silent, wav_path, preview, total_s)
                os.unlink(silent)
            except Exception as exc:  # noqa: BLE001
                diagnostics.append({"category": "speech_failure", "phrase": "(audio mux)", "reason": str(exc),
                                    "corrective": "The preview plays without sound."})
                os.replace(silent, preview)
        else:
            os.replace(silent, preview)
        schedule["video"] = "preview.mp4"
    schedule["stages"] = stages
    schedule["prepare_s"] = round(time.time() - t_start, 2)
    schedule["files"] = {"npz": os.path.basename(npz_path), "csv": os.path.basename(csv_path),
                         "wav": "speech.wav" if speech_ok else None, "video": schedule.get("video"), "schedule": "schedule.json"}
    with open(os.path.join(out_dir, "schedule.json"), "w") as handle:
        json.dump(schedule, handle, indent=1, ensure_ascii=False)
    if os.path.exists(raw_mp3):
        os.unlink(raw_mp3)
    return schedule


def export_fallbacks(out_root: str | None = None) -> list[dict]:
    """Write the two fallback motions as library entries (FALLBACK_01, FALLBACK_02).

    They are *fallbacks*, not semantic gestures: FALLBACK_01 is the relaxed
    talking idle (attention -> talking pose -> idle -> attention) and FALLBACK_02
    the attention hold.  They deliberately sit outside the PDF_ number space,
    which belongs to the gesture PDFs (1-250 and 251-500).  Both go through the same validation as a presentation
    and are filed under outputs/web/presentation_fallbacks so the gallery lists
    them without touching gesture_library_250.json.
    """
    out_root = out_root or os.path.join(ROOT, "outputs", "web", "presentation_fallbacks")
    allowlist = ps.build_allowlist()
    attention, att_info = attention_pose(allowlist)
    idle = IdleTrack(attention)
    fps_i = int(FPS)
    specs = []
    # FALLBACK_01: talking idle, 8 s.
    bl, hold, ret = int(0.6 * FPS), int(6.0 * FPS), int(0.9 * FPS)
    t1 = np.concatenate([
        np.repeat(attention[None, :], int(0.4 * FPS), axis=0),
        hermite(attention, np.zeros(36), idle.pose(int(0.4 * FPS) + bl), idle.vel(int(0.4 * FPS) + bl), bl),
        idle.block(int(0.4 * FPS) + bl, int(0.4 * FPS) + bl + hold),
        hermite(idle.pose(int(0.4 * FPS) + bl + hold), idle.vel(int(0.4 * FPS) + bl + hold), attention, np.zeros(36), ret),
        np.repeat(attention[None, :], int(0.4 * FPS), axis=0)])
    tr1 = [{"kind": "to_talking", "start_f": int(0.4 * FPS), "end_f": int(0.4 * FPS) + bl, "label": "attention -> talking"},
           {"kind": "return", "start_f": int(0.4 * FPS) + bl + hold, "end_f": int(0.4 * FPS) + bl + hold + ret, "label": "return"}]
    specs.append(("FALLBACK_01", None, "Talking Idle (fallback)", "Relaxed talking pose with slow conversational forearm and hand "
                  "movement and still moments. Fallback motion used between gestures in presentation mode; not a semantic gesture.", t1, tr1))
    # FALLBACK_02: attention hold, 2 s.
    t2 = np.repeat(attention[None, :], 2 * fps_i, axis=0)
    specs.append(("FALLBACK_02", None, "Attention Hold (fallback)", "Verified standing attention pose held still. Start and end pose "
                  "of every presentation; not a semantic gesture.", t2, []))
    written = []
    for gid, number, intent, prompt, qpos, trans in specs:
        d = os.path.join(out_root, gid)
        os.makedirs(d, exist_ok=True)
        csv_path = os.path.join(d, f"{gid}.csv")
        np.savetxt(csv_path, qpos, delimiter=",")
        np.savez_compressed(os.path.join(d, f"{gid}.npz"), qpos=qpos.astype(np.float32), fps=np.asarray(FPS, dtype=np.float32))
        check = ps.check_timeline(qpos, trans)
        val = _run_validator(csv_path)
        ok = check["ok"] and val["returncode"] == 0
        if ok:
            render_paced(qpos, os.path.join(d, f"{gid}.mp4"))
        meta = {"name": gid, "number": number, "intent": intent, "prompt": prompt, "collection": "presentation_fallbacks",
                "category": "Presentation fallbacks", "fallback": True, "frames": int(len(qpos)), "fps": FPS,
                "duration_s": round(len(qpos) / FPS, 3), "clamped": True, "valid": ok,
                "qpos_shape": [1, int(len(qpos)), 36], "device": "cpu", "generation_s": 0.0,
                "report": "\n".join(l for l in val["stdout"].splitlines() if not l.startswith("=") and l.strip()),
                "stationary_check": {k: check[k] for k in ("ok", "reasons", "warnings", "measurements", "transitions")},
                "attention_pose": att_info["source"], "talking_pose_arms": TALKING_ARMS,
                "video": f"/files/presentation_fallbacks/{gid}/{gid}.mp4", "csv": f"/files/presentation_fallbacks/{gid}/{gid}.csv",
                "npz": f"/files/presentation_fallbacks/{gid}/{gid}.npz"}
        with open(os.path.join(d, "meta.json"), "w") as handle:
            json.dump(meta, handle, indent=2)
        written.append({"gesture_id": gid, "valid": ok, "reasons": check["reasons"], "dir": d})
    return written


def _run_validator(csv_path: str) -> dict:
    val_script = os.path.join(ROOT, "src", "validate_g1_motion.py")
    if not os.path.isfile(val_script):
        val_script = os.path.join(ROOT, "validate_g1_motion.py")
    proc = subprocess.run([sys.executable, val_script, csv_path],
                          cwd=ROOT, capture_output=True, text=True, timeout=300)
    return {"returncode": proc.returncode, "stdout": proc.stdout[-4000:], "stderr": proc.stderr[-1000:]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", nargs="?", default="")
    ap.add_argument("--out", default=None)
    ap.add_argument("--voice", default="en-US-AvaNeural")
    ap.add_argument("--wpm", type=float, default=150.0)
    ap.add_argument("--phrase-pause", type=float, default=None)
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--export-fallbacks", action="store_true", help="write FALLBACK_01/02 clips and exit")
    args = ap.parse_args()
    if args.export_fallbacks:
        for w in export_fallbacks():
            print(json.dumps(w))
        return 0
    out = args.out or os.path.join(OUT_ROOT, re.sub(r"[^a-z0-9]+", "_", args.text.lower()).strip("_")[:32] or "presentation")
    sched = prepare(args.text, out, args.voice, args.wpm, args.phrase_pause, render=not args.no_render,
                    progress=lambda label, frac: print(f"  [{frac * 100:3.0f}%] {label}", flush=True))
    print(json.dumps({k: sched[k] for k in ("duration_s", "speech_end_s", "gestures", "diagnostics", "pauses")}, indent=1))
    print("valid:", sched["validation"]["ok"], "->", out)
    return 0 if sched["validation"]["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
