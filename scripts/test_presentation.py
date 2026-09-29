#!/usr/bin/env python3
"""Acceptance tests for presentation mode (speech + allowed gestures).

Runs the composer on the required texts and checks the schedule, the measured
motion and -- for the main paragraph -- the rendered MP4's audio against the
schedule.  Edge TTS needs network access; without it the speech cases are
reported as skipped rather than passed.

    python test_presentation.py            # all cases, no video except the main paragraph
    python test_presentation.py --render   # render every case

Exit code 0 only when every check passes.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_CUR_DIR) if os.path.basename(_CUR_DIR) == "scripts" else _CUR_DIR
SRC_DIR = os.path.join(ROOT, "src")
for _p in (ROOT, SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import presentation_compose as pc  # noqa: E402
import presentation_safety as ps  # noqa: E402

OUT = os.path.join(ROOT, "outputs", "presentations", "_tests")
MAIN = ("Hello! Welcome to robotics. Let me explain about our robot. "
        "Our robot can walk, run, jump, and perform a back flip. Thank you.")
RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(cond), detail))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""), flush=True)


def run(name: str, text: str, render: bool = False, **kw) -> dict:
    out = os.path.join(OUT, name)
    shutil.rmtree(out, ignore_errors=True)
    t0 = time.time()
    s = pc.prepare(text, out, render=render, name=name, **kw)
    print(f"\n== {name}: {time.time() - t0:.1f}s, {len(s['gestures'])} gestures, {len(s['diagnostics'])} diagnostics, valid={s['validation']['ok']}")
    for g in s["gestures"]:
        print(f"     {g['intent']:12s} '{g['phrase']}' word {g['word_start_s']:.2f} main {g['main_start_s']:.2f}-{g['main_end_s']:.2f} lag {g['main_lag_s'] * 1000:+.0f} ms")
    for d in s["diagnostics"]:
        print(f"     ! {d['category']:20s} '{d['phrase']}': {d['reason'][:110]}")
    return s


def gestures_by_phrase(s: dict) -> dict:
    return {g["phrase"].lower(): g for g in s["gestures"]}


def diags(s: dict, category: str) -> list[dict]:
    return [d for d in s["diagnostics"] if d["category"] == category]


def measure_audio_sync(out_dir: str, s: dict) -> tuple[float, int]:
    """Max |scheduled word start - audible onset| in the muxed MP4, for words after a silence."""
    pcm, dur, sr = pc.decode_audio(os.path.join(out_dir, "preview.mp4"))
    win = int(0.005 * sr)
    env = np.convolve(np.abs(pcm), np.ones(win) / win, mode="same")
    thr = env.max() * 0.03
    errs = []
    for w in s["words"]:
        i0 = int(w["start_s"] * sr)
        pre = env[max(0, i0 - int(0.12 * sr)): i0]
        if pre.size and pre.max() > thr:
            continue
        a = max(0, i0 - int(0.1 * sr))
        seg = env[a: i0 + int(0.3 * sr)]
        if not (seg > thr).any():
            continue
        errs.append(abs((a + np.argmax(seg > thr)) / sr - w["start_s"]) * 1000)
    return (max(errs) if errs else float("nan")), len(errs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    al = ps.build_allowlist()

    # ------------------------------------------------------------ allowlist
    print("\n== allowlist")
    check("Goodbye (PDF_009) is denied despite its name", not al["clips"]["PDF_009"]["allowed"],
          "; ".join(al["clips"]["PDF_009"]["reasons"])[:100])
    check("Walk/Run/Kick/Sit denied", all(not al["clips"][g]["allowed"] for g in ("PDF_090", "PDF_091", "PDF_192", "PDF_088")))
    check("Hello/Welcome/Explain/Thank You allowed", all(al["clips"][g]["allowed"] for g in ("PDF_001", "PDF_007", "PDF_209", "PDF_014")))
    check("every allowed clip has feet planted (lift<=1.5cm, slide<=3cm, drift<=6cm)",
          all(c["measurements"]["foot_lift_m"] <= 0.015 and c["measurements"]["foot_slide_m"] <= 0.03
              and c["measurements"]["root_drift_m"] <= 0.06 for c in al["clips"].values() if c["allowed"]))

    # ------------------------------------------------------------ main paragraph
    s = run("main", MAIN, render=True)
    speech = s["speech_ok"]
    g = gestures_by_phrase(s)
    check("Hello -> allowed wave PDF_001", g.get("hello", {}).get("gesture_id") == "PDF_001")
    if speech:
        check("wave main action aligned to 'Hello' (|lag| <= 250 ms)", abs(g.get("hello", {}).get("main_lag_s", 9)) <= 0.25,
              f"lag {g.get('hello', {}).get('main_lag_s', 9) * 1000:+.0f} ms")
    check("Welcome -> allowed welcoming gesture PDF_007", any(x["gesture_id"] == "PDF_007" for x in s["gestures"]))
    check("explanation -> allowed Explain gesture or talking fallback",
          any(x["gesture_id"] == "PDF_209" for x in s["gestures"]) or any(seg["kind"] == "talking" for seg in s["segments"]))
    blocked = {d["phrase"].lower() for d in diags(s, "blocked_restricted")}
    check("walk/run/jump/back flip blocked as spoken descriptions", {"walk", "run", "jump", "back flip"} <= blocked, str(sorted(blocked)))
    check("no restricted clip in the schedule", all(al["clips"][x["gesture_id"]]["allowed"] for x in s["gestures"]))
    m = s["validation"]["stationary_and_continuity"]["measurements"]
    check("feet planted / no location change in composed output",
          m["foot_lift_m"] <= 0.015 and m["foot_slide_m"] <= 0.03 and m["root_drift_m"] <= 0.06,
          f"lift {m['foot_lift_m']:.3f} slide {m['foot_slide_m']:.3f} drift {m['root_drift_m']:.3f} m")
    check("Thank you -> allowed PDF_014", g.get("thank you", {}).get("gesture_id") == "PDF_014")
    q = np.load(os.path.join(OUT, "main", "main.npz"))["qpos"]
    att = q[0]
    # Only what lies strictly between two gestures counts: the first entry
    # legitimately starts from attention and the final return ends there.
    gsegs = [seg for seg in s["segments"] if seg["kind"] == "gesture"]
    lo, hi = (gsegs[0]["end_f"], gsegs[-1]["start_f"]) if len(gsegs) >= 2 else (0, 0)
    between = [seg for seg in s["segments"] if seg["kind"] in ("talking", "transition", "exit", "entry")
               and seg["start_f"] >= lo and seg["end_f"] <= hi]
    dips = 0
    for seg in between:
        block = q[seg["start_f"]: seg["end_f"]]
        if len(block) and np.min(np.linalg.norm(block[:, 22:36] - att[22:36], axis=1)) < 0.15:
            dips += 1
    check("hands do not return to attention between phrases", dips == 0, f"{dips} inter-phrase segment(s) reached the attention arm pose")
    check("presentation ends in attention pose", np.allclose(q[-1], att, atol=1e-6) and s["segments"][-1]["kind"] == "attention")
    check("starts with attention hold", s["segments"][0]["kind"] == "attention" and s["segments"][0]["end_s"] >= 0.5)
    check("all checks valid (stationary, continuity, validate_g1_motion.py)", s["validation"]["ok"])
    if speech:
        err, n = measure_audio_sync(os.path.join(OUT, "main"), s)
        check("muxed audio onsets match schedule (<= 40 ms)", err <= 40, f"max {err:.1f} ms over {n} onsets")
        check("gesture main action lag within +250/-100 ms", all(-0.1 <= x["main_lag_s"] <= 0.25 for x in s["gestures"]),
              "max |lag| %.0f ms" % (max(abs(x["main_lag_s"]) for x in s["gestures"]) * 1000))
    else:
        check("speech synthesized (network)", False, "Edge TTS unavailable -- speech checks skipped")

    # ------------------------------------------------------------ no matching gestures
    s = run("nomatch", "The quarterly numbers were reviewed by the committee on Tuesday.", render=args.render)
    check("no gestures scheduled", len(s["gestures"]) == 0)
    check("talking fallback used", any(seg["kind"] == "talking" for seg in s["segments"]))
    check("no_gesture diagnostic present", bool(diags(s, "no_gesture")))
    check("valid", s["validation"]["ok"])

    # ------------------------------------------------------------ two gestures close together
    s = run("close", "Thank you, welcome everyone.", render=args.render)
    check("both gestures scheduled with a pause on the comma", len(s["gestures"]) == 2 and any("welcome" in p["label"].lower() for p in s["pauses"]),
          str([(p["label"], p["duration_s"]) for p in s["pauses"]]))
    if len(s["gestures"]) == 2:
        a, b = s["gestures"]
        check("first main action played whole before the second", b["main_start_s"] >= a["main_end_s"] + 0.4)
        check("second main action still on its word", abs(b["main_lag_s"]) <= 0.25, f"lag {b['main_lag_s'] * 1000:+.0f} ms")
    check("valid", s["validation"]["ok"])
    s2 = run("close_nopunct", "Thank you welcome everyone", render=False)
    check("without punctuation: conflict reported, gesture spoken only (not cut short)",
          len(s2["gestures"]) == 1 and bool(diags(s2, "timing_conflict")))

    # ------------------------------------------------------------ negation / restricted instruction
    s = run("negation", "Do not jump. I do not think it is dangerous.", render=args.render)
    check("'jump' blocked (restricted), nothing scheduled for it", "jump" in {d["phrase"].lower() for d in diags(s, "blocked_restricted")}
          and not any("jump" in x["phrase"].lower() for x in s["gestures"]))
    check("negated 'think' not gestured", any(d["status"] == "negated" for d in s["diagnostics"]) and not any(x["intent"] == "Think" for x in s["gestures"]))
    s = run("command", "Walk forward now.", render=args.render)
    check("'walk forward' blocked, no locomotion clip scheduled", bool(diags(s, "blocked_restricted")) and len(s["gestures"]) == 0
          or all(al["clips"][x["gesture_id"]]["allowed"] and al["clips"][x["gesture_id"]]["restricted_class"] is None for x in s["gestures"]),
          str([(x["intent"]) for x in s["gestures"]]))
    check("feet planted", s["validation"]["stationary_and_continuity"]["measurements"]["foot_slide_m"] <= 0.03)

    # ------------------------------------------------------------ misleadingly named clip
    s = run("misleading", "Wave goodbye to everyone. Goodbye!", render=args.render)
    check("'wave goodbye' -> PDF_009 rejected by measured motion", any(d["gesture_id"] == "PDF_009" and d["category"] == "blocked_allowlist" for d in s["diagnostics"]),
          "; ".join(d["reason"][:90] for d in s["diagnostics"] if d.get("gesture_id") == "PDF_009"))
    check("'Goodbye!' uses the allowed stationary wave PDF_010", any(x["gesture_id"] == "PDF_010" for x in s["gestures"]))

    # ------------------------------------------------------------ quoted action
    s = run("quoted", 'The manual says "please be careful" on every page.', render=False)
    check("quoted phrases spoken without gesture", any(d["status"] == "quoted" for d in s["diagnostics"]) and len(s["gestures"]) == 0)

    # ------------------------------------------------------------ other voice / rate
    s = run("voice", "Hello everyone. Thank you.", render=args.render, voice="en-US-AndrewNeural", words_per_minute=120, phrase_pause_s=1.5)
    check("different voice + rate rebuilt (rate -20%)", s["voice"] == "en-US-AndrewNeural" and s["tts_rate"] == "-20%")
    check("phrase pause applied (>= 1.5 s at sentence end)", any(p["duration_s"] >= 1.5 for p in s["pauses"] if "sentence" in p["label"]))
    if s["speech_ok"]:
        check("gestures aligned on the slower voice", all(abs(x["main_lag_s"]) <= 0.25 for x in s["gestures"]) and len(s["gestures"]) == 2)

    # ------------------------------------------------------------ missing asset
    victim = os.path.join(ps.LIBRARY_250_OUTPUT, "PDF_210", "PDF_210.csv")
    backup = victim + ".bak_test"
    try:
        os.rename(victim, backup)
        s = run("missing", "Any questions? Thank you.", render=False)
        check("missing CSV reported as asset_missing, presentation still valid",
              any(d["category"] == "asset_missing" and d.get("gesture_id") == "PDF_210" for d in s["diagnostics"]) and s["validation"]["ok"])
    finally:
        if os.path.exists(backup):
            os.rename(backup, victim)
        ps.build_allowlist()
    check("allowlist restored after the test", ps.build_allowlist()["clips"]["PDF_210"]["allowed"])

    # ------------------------------------------------------------ fallback clips
    for gid in ("FALLBACK_01", "FALLBACK_02"):
        meta_path = os.path.join(ROOT, "outputs", "web", "presentation_fallbacks", gid, "meta.json")
        ok = os.path.isfile(meta_path) and json.load(open(meta_path)).get("valid")
        check(f"{gid} fallback clip exists and validated", bool(ok))

    print("\n== summary")
    failed = [r for r in RESULTS if not r[1]]
    print(f"  {len(RESULTS) - len(failed)}/{len(RESULTS)} passed")
    for name, _, detail in failed:
        print(f"  FAILED: {name} {detail}")
    with open(os.path.join(OUT, "results.json"), "w") as handle:
        json.dump([{"check": n, "pass": p, "detail": d} for n, p, d in RESULTS], handle, indent=1)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
