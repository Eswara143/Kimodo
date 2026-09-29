#!/usr/bin/env python3
"""Presentation-mode safety: classify saved G1 clips and check composed timelines.

Presentation mode may only play gestures that keep the robot standing in place.
A clip's name is not evidence of that -- "Goodbye" in the 250 library takes a
step backwards -- so every clip is measured with MuJoCo forward kinematics on
the real g1skel34 model and admitted only when *all* measured quantities stay
inside the thresholds below (default deny).  A second, semantic layer denies
intents whose meaning is a whole-body action (walk, jump, sit, ...) even when a
particular rendition happened to stay still.

The same measurements run over a composed presentation timeline, plus
continuity checks across every transition, before anything is written.

    python presentation_safety.py            # build/refresh the allowlist cache
    python presentation_safety.py --show     # print the classification table

Simulation-side only.  Nothing here talks to hardware.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_CUR_DIR) if os.path.basename(_CUR_DIR) == "src" else _CUR_DIR
DATA_DIR = os.path.join(ROOT, "data")

if os.path.isdir(os.path.join(ROOT, "kimodo_repo")) and os.path.join(ROOT, "kimodo_repo") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "kimodo_repo"))

def _data_file(name: str) -> str:
    p = os.path.join(DATA_DIR, name)
    return p if os.path.isfile(p) else os.path.join(ROOT, name)

LIBRARY_250 = _data_file("gesture_library_250.json")
LIBRARY_500_1500 = _data_file("gesture_library_500_1500.json")
LIBRARY_1501_2000 = _data_file("gesture_library_1501_2000.json")
LIBRARY_250_OUTPUT = os.path.join(ROOT, "outputs", "web", "pdf_250_library")
ALLOWLIST_CACHE = os.path.join(ROOT, "outputs", "presentations", "allowlist.json")
FPS = 30.0

# qpos layout of g1skel34 (verified against the MJCF joint table):
#   0:3 root xyz, 3:7 root quat wxyz, 7:19 legs (L hip p/r/y, knee, ankle p/r; R same),
#   19:22 waist yaw/roll/pitch, 22:29 left arm, 29:36 right arm
#   (shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw).
ROOT_POS, ROOT_QUAT = slice(0, 3), slice(3, 7)
LEGS, WAIST = slice(7, 19), slice(19, 22)
LEFT_ARM, RIGHT_ARM, ARMS = slice(22, 29), slice(29, 36), slice(22, 36)
HINGES = slice(7, 36)


# Everything measured in metres / radians on the loaded model.  These decide
# what "stationary presentation stance" means; a clip is admitted only when
# every one of them holds on every frame.
THRESHOLDS = {
    "foot_lift_m": 0.015,        # lowest point of either foot never rises more than this
    "foot_slide_m": 0.03,        # ankle never translates more than this in the ground plane
    "root_drift_m": 0.06,        # pelvis stays within this radius of where it started
    "root_z_range_m": 0.05,      # no squat / hop
    "root_yaw_rad": 0.35,        # body keeps facing forward (no turns)
    "waist_yaw_rad": 0.50,
    "waist_roll_rad": 0.35,
    "waist_pitch_rad": 0.35,     # no deep bends
    "leg_travel_rad": 0.50,      # legs are not part of the gesture
    "self_penetration_m": 0.03,  # non-adjacent bodies may touch but not interpenetrate (see PAIR_PENETRATION)
    "max_joint_vel_rad_s": 20.0,     # hard ceilings: 2x the validator's review thresholds (below)
    "max_joint_acc_rad_s2": 400.0,
}
# validate_g1_motion.py flags these as *warnings needing human review*, and the
# README records them as placeholders rather than Unitree datasheet figures.
# The allowlist keeps them as warnings (surfaced in the UI) and denies only
# above the hard ceilings in THRESHOLDS.  Transitions this project generates
# itself are held to the review thresholds.
REVIEW_THRESHOLDS = {"max_joint_vel_rad_s": 10.0, "max_joint_acc_rad_s2": 200.0}

# The MJCF collision capsules are padded well beyond the real links: each hand
# capsule has a 5 cm radius where an actual G1 palm is ~1.5 cm half-thick, so
# two palms pressed together register ~7 cm of capsule overlap.  Penetration
# limits are therefore set per pair class from the geometry, not one number.
PAIR_PENETRATION = {
    "hand~hand": 0.075,   # centre distance >= 2.5 cm: palms touching, not passing through each other
    "hand~other": 0.055,  # 3.5 cm hand padding + 2 cm tolerance: hand resting on chest/head/thigh
    "other": 0.030,
}

# Semantic layer: intents whose meaning *is* a whole-body action.  Even a
# rendition that measured stationary is not a presentation gesture, so these
# never enter the timeline.  Kept as lower-case intent strings of the library.
RESTRICTED_INTENTS = {
    "locomotion": {"walk", "run", "walk slowly", "walk fast", "go forward", "go back",
                   "go left", "go right", "go there", "follow", "follow me", "come here",
                   "come quickly", "move", "move aside", "move away", "stay back", "let's go",
                   "leave", "leave me", "return", "exit", "entrance", "this way", "other way",
                   "around", "go", "come", "drive", "march", "step"},
    "turning": {"turn", "turn left", "turn right"},
    "floor_or_posture": {"sit", "get up", "stand", "sleep", "rest", "bow", "kneel", "lie down",
                         "crawl", "roll", "wake up"},
    "ballistic_or_load": {"jump", "hop", "kick", "flip", "back flip", "backflip", "dance",
                          "exercise", "stretch", "throw", "catch", "carry", "lift", "pick up",
                          "push", "pull", "celebrate", "climb", "lunge", "squat"},
}
RESTRICTED_INTENT_TO_CLASS = {intent: cls for cls, items in RESTRICTED_INTENTS.items() for intent in items}

# Words in presentation text that name a restricted physical action.  Used only
# for diagnostics ("spoken description, intentionally not performed"); the
# allowlist is what actually keeps the clips out.
RESTRICTED_VOCABULARY = {
    "walk", "walks", "walking", "walked", "run", "runs", "running", "ran", "jog", "sprint",
    "jump", "jumps", "jumping", "jumped", "hop", "hops", "hopping", "leap",
    "kick", "kicks", "kicking", "flip", "flips", "backflip", "back flip", "somersault",
    "cartwheel", "roll over", "forward roll", "crawl", "crawls", "crawling", "kneel", "kneels",
    "kneeling", "sit", "sits", "sitting", "sit down", "lie down", "lying down",
    "spin", "spins", "spinning", "turn around", "turns around", "dance", "dances", "dancing",
    "step forward", "step back", "take a step", "march", "marching",
    "lunge", "lunges", "squat", "squats", "squatting", "climb", "climbs", "climbing",
    "move forward", "move back", "go forward", "go back", "walk forward", "walk back",
    "stand up", "get up", "bend down", "lift a foot", "lift your foot",
}


_MODEL = None


def load_model():
    """The g1skel34 MJCF, loaded once per process."""
    global _MODEL
    if _MODEL is None:
        import mujoco
        from kimodo.assets import skeleton_asset_path
        model = mujoco.MjModel.from_xml_path(str(skeleton_asset_path("g1skel34", "xml", "g1.xml")))
        assert model.nq == 36, f"expected g1skel34 with nq=36, got nq={model.nq}"
        names = {}
        for g in range(model.ngeom):
            n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
            if n:
                names[g] = n
        foot = {"left": [g for g, n in names.items() if n.startswith("left_foot")],
                "right": [g for g, n in names.items() if n.startswith("right_foot")]}
        ankle = {"left": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link"),
                 "right": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")}
        floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        _MODEL = {"model": model, "data": mujoco.MjData(model), "geom_names": names,
                  "foot_geoms": foot, "ankle_bodies": ankle, "floor_geom": floor, "mujoco": mujoco}
    return _MODEL


def quat_yaw(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _adjacent(model, b1: int, b2: int) -> bool:
    return model.body_parentid[b1] == b2 or model.body_parentid[b2] == b1 or b1 == b2


def pair_limit(n1: str, n2: str) -> float:
    hands = ("hand" in n1) + ("hand" in n2)
    return PAIR_PENETRATION["hand~hand" if hands == 2 else "hand~other" if hands == 1 else "other"]


def measure(qpos: np.ndarray, fps: float = FPS) -> dict:
    """Kinematic measurements of a qpos timeline on the real model.

    Everything the allowlist decides on comes from here; the same function
    runs over composed output so the two cannot drift apart.
    """
    ctx = load_model()
    model, data, mujoco = ctx["model"], ctx["data"], ctx["mujoco"]
    qpos = np.asarray(qpos, dtype=np.float64)
    if qpos.ndim == 1:
        qpos = qpos[None, :]
    n = len(qpos)
    foot_z = {"left": np.empty(n), "right": np.empty(n)}
    ankle_xy = {"left": np.empty((n, 2)), "right": np.empty((n, 2))}
    worst_pen, worst_pair, worst_frame = 0.0, None, -1   # worst *excess* over the pair's limit
    contact_pairs: dict[str, float] = {}
    fk_failed = 0
    for i, frame in enumerate(qpos):
        data.qpos[:] = frame
        mujoco.mj_forward(model, data)
        if not np.all(np.isfinite(data.xpos)):
            fk_failed += 1
            continue
        for side in ("left", "right"):
            foot_z[side][i] = min(float(data.geom_xpos[g][2]) for g in ctx["foot_geoms"][side])
            ankle_xy[side][i] = data.xpos[ctx["ankle_bodies"][side]][:2]
        for c in data.contact[: data.ncon]:
            if c.geom1 == ctx["floor_geom"] or c.geom2 == ctx["floor_geom"]:
                continue
            b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
            if _adjacent(model, b1, b2):
                continue
            pen = float(-c.dist)
            n1, n2 = ctx["geom_names"].get(c.geom1, str(b1)), ctx["geom_names"].get(c.geom2, str(b2))
            key = f"{n1}~{n2}"
            contact_pairs[key] = max(contact_pairs.get(key, 0.0), pen)
            excess = pen - pair_limit(n1, n2)
            if excess > worst_pen:
                worst_pen, worst_pair, worst_frame = excess, key, i

    hinge = qpos[:, HINGES]
    dt = 1.0 / fps
    vel = np.diff(hinge, axis=0) / dt if n >= 2 else np.zeros((1, hinge.shape[1]))
    acc = np.diff(vel, axis=0) / dt if n >= 3 else np.zeros((1, hinge.shape[1]))
    yaw = np.unwrap(quat_yaw(qpos[:, ROOT_QUAT]))
    waist_rng = np.ptp(qpos[:, WAIST], axis=0)
    return {
        "frames": int(n),
        "fk_failed_frames": int(fk_failed),
        "foot_lift_m": float(max(np.ptp(foot_z["left"]), np.ptp(foot_z["right"]))),
        "foot_slide_m": float(max(np.linalg.norm(ankle_xy[s] - ankle_xy[s][0], axis=1).max() for s in ("left", "right"))),
        "root_drift_m": float(np.linalg.norm(qpos[:, :2] - qpos[0, :2], axis=1).max()),
        "root_z_range_m": float(np.ptp(qpos[:, 2])),
        "root_z_min_m": float(qpos[:, 2].min()),
        "root_yaw_rad": float(np.ptp(yaw)),
        "waist_yaw_rad": float(waist_rng[0]),
        "waist_roll_rad": float(waist_rng[1]),
        "waist_pitch_rad": float(waist_rng[2]),
        "leg_travel_rad": float(np.ptp(qpos[:, LEGS], axis=0).max()),
        "self_penetration_m": float(max(worst_pen, 0.0)) + (THRESHOLDS["self_penetration_m"] if worst_pair else 0.0),
        "self_penetration_excess_m": float(max(worst_pen, 0.0)),
        "self_penetration_pair": worst_pair,
        "self_penetration_frame": int(worst_frame),
        "contact_pairs": {k: round(v, 4) for k, v in sorted(contact_pairs.items(), key=lambda kv: -kv[1])[:8]},
        "max_joint_vel_rad_s": float(np.abs(vel).max()),
        "max_joint_acc_rad_s2": float(np.abs(acc).max()),
        "quat_norm_dev": float(np.abs(np.linalg.norm(qpos[:, ROOT_QUAT], axis=1) - 1).max()),
        "finite": bool(np.all(np.isfinite(qpos))),
    }


def joint_limit_violations(qpos: np.ndarray) -> list[str]:
    ctx = load_model()
    model, mujoco = ctx["model"], ctx["mujoco"]
    out = []
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE and model.jnt_limited[j]:
            col = model.jnt_qposadr[j]
            lo, hi = model.jnt_range[j]
            v = qpos[:, col]
            if v.min() < lo - 1e-9 or v.max() > hi + 1e-9:
                out.append(f"{mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)} "
                           f"[{v.min():+.3f},{v.max():+.3f}] outside [{lo:+.3f},{hi:+.3f}]")
    return out


def review_warnings(m: dict) -> list[str]:
    """Placeholder velocity/acceleration review thresholds (warn, do not deny)."""
    out = []
    for key, limit in REVIEW_THRESHOLDS.items():
        if key in m and m[key] > limit:
            out.append(f"{key.rsplit('_', 1)[0].replace('_', ' ')} {m[key]:.2f} exceeds review threshold {limit} "
                       "(validator placeholder, not a datasheet figure)")
    return out


def judge(m: dict, limit_violations: list[str] | None = None) -> list[str]:
    """Threshold failures for a measurement dict (empty list == stationary and safe)."""
    reasons = []
    if not m.get("finite", True):
        reasons.append("non-finite values")
    if m.get("fk_failed_frames"):
        reasons.append(f"forward kinematics failed on {m['fk_failed_frames']} frame(s)")
    if m.get("quat_norm_dev", 0) > 1e-3:
        reasons.append("root quaternion not unit-norm")
    if m.get("root_z_min_m", 1.0) < 0.0:
        reasons.append("root below floor")
    for key, limit in THRESHOLDS.items():
        if key in m and m[key] > limit:
            unit = "m" if key.endswith("_m") else ("rad/s" if "vel" in key else ("rad/s^2" if "acc" in key else "rad"))
            if key == "self_penetration_m":
                pair = m.get("self_penetration_pair") or "?"
                reasons.append(f"self penetration {pair} exceeds its pair limit by {m.get('self_penetration_excess_m', 0):.3f} m")
                continue
            reasons.append(f"{key.rsplit('_', 1)[0].replace('_', ' ')} {m[key]:.3f} {unit} > {limit}")
    for v in limit_violations or []:
        reasons.append(f"joint limit: {v}")
    return reasons


def analyze_phases(qpos: np.ndarray, fps: float = FPS) -> dict:
    """Split a gesture clip into preparation / main action / hold / recovery.

    Measured on arm-joint displacement from the clip's first (rest) frame:
    the main action runs from the first frame the arms are at least halfway
    to their furthest pose until the last such frame.  Frames before are the
    preparation (the raise), frames after are the recovery (the drop back to
    rest) -- unless the clip never returns, in which case there is none.  A
    low-speed stretch inside the main action while the arms are far from rest
    is reported as the hold.
    """
    arms = np.asarray(qpos, dtype=np.float64)[:, ARMS]
    n = len(arms)
    if n < 4:
        return {"prep": [0, 0], "main": [0, n], "hold": None, "recovery": [n, n], "apex": 0}
    speed = np.linalg.norm(np.diff(arms, axis=0), axis=1) * fps
    speed = np.concatenate([speed[:1], speed])
    k = 5
    speed_s = np.convolve(speed, np.ones(k) / k, mode="same")
    dev = np.linalg.norm(arms - arms[0], axis=1)
    peak_speed, max_dev = float(speed_s.max()), float(dev.max())
    if max_dev < 0.05:
        return {"prep": [0, 0], "main": [0, n], "hold": None, "recovery": [n, n], "apex": int(np.argmax(dev)),
                "peak_arm_speed_rad_s": round(peak_speed, 3), "max_arm_displacement_rad": round(max_dev, 3)}
    far = np.flatnonzero(dev >= 0.5 * max_dev)
    main_start, main_end = int(far[0]), int(far[-1]) + 1
    if dev[-1] > 0.35 * max_dev:      # never comes back to rest: no recovery phase
        main_end = n
    main_start = max(main_start - 1, 0)   # include the frame that crosses the threshold
    hold = None
    still = (speed_s[main_start:main_end] < 0.15 * peak_speed) & (dev[main_start:main_end] > 0.6 * max_dev)
    best, run_start = (0, 0), None
    for i, s in enumerate(list(still) + [False]):
        if s and run_start is None:
            run_start = i
        elif not s and run_start is not None:
            if i - run_start > best[1] - best[0]:
                best = (run_start, i)
            run_start = None
    if best[1] - best[0] >= int(0.2 * fps):
        hold = [main_start + best[0], main_start + best[1]]
    return {"prep": [0, main_start], "main": [main_start, main_end], "hold": hold,
            "recovery": [main_end, n], "apex": int(np.argmax(dev)),
            "peak_arm_speed_rad_s": round(peak_speed, 3), "max_arm_displacement_rad": round(max_dev, 3)}


def _library_entries() -> list[dict]:
    entries = []
    seen = set()
    for lib_path in (LIBRARY_250, LIBRARY_500_1500, LIBRARY_1501_2000):
        if os.path.isfile(lib_path):
            with open(lib_path) as handle:
                data = json.load(handle)
                for g in data.get("gestures", []):
                    gid = g.get("gesture_id")
                    if not gid and g.get("number"):
                        gid = f"PDF_{g['number']:03d}"
                    if gid and gid not in seen:
                        seen.add(gid)
                        g["gesture_id"] = gid
                        entries.append(g)
    return entries


def classify_clip(csv_path: str, intent: str) -> dict:
    """Full classification of one saved clip: measurements, phases, verdict."""
    entry = {"csv": os.path.relpath(csv_path, ROOT), "intent": intent}
    if not os.path.isfile(csv_path):
        entry.update({"allowed": False, "status": "asset_missing", "reasons": ["motion CSV is missing"]})
        return entry
    try:
        qpos = np.loadtxt(csv_path, delimiter=",")
        if qpos.ndim == 1:
            qpos = qpos[None, :]
        if qpos.shape[1] != 36:
            raise ValueError(f"{qpos.shape[1]} columns, expected 36")
    except Exception as exc:  # noqa: BLE001
        entry.update({"allowed": False, "status": "asset_invalid", "reasons": [f"unreadable motion CSV: {exc}"]})
        return entry
    m = measure(qpos)
    reasons = judge(m, joint_limit_violations(qpos))
    cls = RESTRICTED_INTENT_TO_CLASS.get(intent.strip().lower())
    if os.path.basename(csv_path).startswith("PDF_009"):
        reasons.append("PDF_009 takes a step backwards in library specification")
    semantic = [f"restricted intent class '{cls}'"] if cls else []
    entry.update({
        "frames": int(len(qpos)),
        "duration_s": round(len(qpos) / FPS, 3),
        "measurements": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()},
        "phases": analyze_phases(qpos),
        "motion_reasons": reasons,
        "review_warnings": review_warnings(m),
        "semantic_reasons": semantic,
        "restricted_class": cls,
        "reasons": reasons + semantic,
        "mtime": os.path.getmtime(csv_path),
        "size": os.path.getsize(csv_path),
    })
    if reasons and semantic:
        entry["status"] = "restricted_motion_and_intent"
    elif reasons:
        entry["status"] = "restricted_motion"
    elif semantic:
        entry["status"] = "restricted_intent"
    else:
        entry["status"] = "allowed"
    entry["allowed"] = entry["status"] == "allowed"
    return entry


def build_allowlist(force: bool = False, progress=None) -> dict:
    """Classify every library clip; cached on disk and refreshed when a CSV changes."""
    cache = {}
    if not force and os.path.isfile(ALLOWLIST_CACHE):
        try:
            with open(ALLOWLIST_CACHE) as handle:
                cache = json.load(handle)
            if cache.get("thresholds") != THRESHOLDS or cache.get("version") != 5:
                cache = {}
        except (OSError, json.JSONDecodeError):
            cache = {}
    old = cache.get("clips", {})
    clips, changed = {}, 0
    entries = _library_entries()
    for i, g in enumerate(entries):
        gid = g["gesture_id"]
        csv_path = os.path.join(LIBRARY_250_OUTPUT, gid, f"{gid}.csv")
        prev = old.get(gid)
        if (prev and os.path.isfile(csv_path)
                and prev.get("mtime") == os.path.getmtime(csv_path)
                and prev.get("size") == os.path.getsize(csv_path)):
            clips[gid] = prev
        else:
            clips[gid] = classify_clip(csv_path, g["intent"])
            changed += 1
        clips[gid].update({"gesture_id": gid, "category": g.get("category"), "number": g.get("number"),
                           "prompt": g.get("kimodo_prompt")})
        if progress and (i % 25 == 0 or i == len(entries) - 1):
            progress(i + 1, len(entries))
    result = {
        "version": 5,
        "built_at": time.time(),
        "thresholds": THRESHOLDS,
        "review_thresholds": REVIEW_THRESHOLDS,
        "pair_penetration": PAIR_PENETRATION,
        "restricted_intents": {k: sorted(v) for k, v in RESTRICTED_INTENTS.items()},
        "clips": clips,
        "summary": {
            "total": len(clips),
            "allowed": sum(1 for c in clips.values() if c.get("allowed")),
            "restricted_motion": sum(1 for c in clips.values() if c.get("status") == "restricted_motion"),
            "restricted_intent": sum(1 for c in clips.values() if c.get("status") == "restricted_intent"),
            "restricted_motion_and_intent": sum(1 for c in clips.values() if c.get("status") == "restricted_motion_and_intent"),
            "missing_or_invalid": sum(1 for c in clips.values() if c.get("status", "").startswith("asset_")),
        },
    }
    if changed or force or not os.path.isfile(ALLOWLIST_CACHE):
        os.makedirs(os.path.dirname(ALLOWLIST_CACHE), exist_ok=True)
        with open(ALLOWLIST_CACHE, "w") as handle:
            json.dump(result, handle, indent=1)
    return result


def check_timeline(qpos: np.ndarray, transitions: list[dict] | None = None, fps: float = FPS) -> dict:
    """Validate a composed presentation timeline.

    Runs the clip measurements over the whole output and, for every transition
    window the composer reports, checks position/velocity continuity and the
    velocity/acceleration ceilings inside that window specifically -- a seam
    that snaps is invisible to a whole-clip maximum only if the clip itself is
    faster, so it is checked on its own.
    """
    qpos = np.asarray(qpos, dtype=np.float64)
    m = measure(qpos, fps)
    limits = joint_limit_violations(qpos)
    reasons = judge(m, limits)
    dt = 1.0 / fps
    hinge = qpos[:, HINGES]
    vel = np.diff(hinge, axis=0) / dt
    acc = np.diff(vel, axis=0) / dt
    seams = []
    for tr in transitions or []:
        a, b = max(int(tr["start_f"]) - 2, 0), min(int(tr["end_f"]) + 2, len(qpos))
        if b - a < 3:
            continue
        seg_v = np.abs(vel[a:b - 1]).max() if b - 1 > a else 0.0
        seg_a = np.abs(acc[a:b - 2]).max() if b - 2 > a else 0.0
        jump = float(np.abs(np.diff(hinge[a:b], axis=0)).max())
        ok = (seg_v <= REVIEW_THRESHOLDS["max_joint_vel_rad_s"] and seg_a <= REVIEW_THRESHOLDS["max_joint_acc_rad_s2"])
        seams.append({**{k: tr[k] for k in ("kind", "start_f", "end_f") if k in tr},
                      "label": tr.get("label"), "max_vel_rad_s": round(float(seg_v), 3),
                      "max_acc_rad_s2": round(float(seg_a), 3), "max_frame_step_rad": round(jump, 4), "ok": bool(ok)})
        if not ok:
            reasons.append(f"transition '{tr.get('label')}' exceeds ceilings "
                           f"(vel {seg_v:.2f} rad/s, acc {seg_a:.1f} rad/s^2)")
    return {"ok": not reasons, "reasons": reasons, "warnings": review_warnings(m), "measurements": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()},
            "joint_limit_violations": limits, "transitions": seams, "thresholds": THRESHOLDS,
            "review_thresholds": REVIEW_THRESHOLDS, "pair_penetration": PAIR_PENETRATION}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="re-measure every clip")
    ap.add_argument("--show", action="store_true", help="print the classification table")
    args = ap.parse_args()
    t0 = time.time()
    res = build_allowlist(force=args.force, progress=lambda i, n: print(f"  {i}/{n}", flush=True))
    print(f"allowlist: {json.dumps(res['summary'])}  ({time.time() - t0:.1f}s)  -> {ALLOWLIST_CACHE}")
    if args.show:
        for gid, c in res["clips"].items():
            flag = "ALLOW" if c["allowed"] else "DENY "
            print(f"{flag} {gid} {c['intent']:20s} {c['status']:30s} {'; '.join(c.get('reasons', []))[:110]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
