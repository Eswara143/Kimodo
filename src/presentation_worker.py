#!/usr/bin/env python3
"""Subprocess entry point for preparing one presentation under resource limits.

webapp.py runs this instead of composing in-process so that the work is bounded
independently of the resident models:

  * host RAM: RLIMIT_DATA soft 4 GiB / hard 5 GiB (RLIMIT_AS cannot be used --
    the NVIDIA EGL driver maps more virtual address space than that and aborts)
  * CPU: BLAS/OpenMP capped at 2 threads, x264 at 2 threads, process niced
  * GPU: the EGL render loop is paced (presentation_compose.render_paced)

No Kimodo model and no text encoder are loaded here.

    python presentation_worker.py --request req.json --out <dir>

Progress lines go to stdout as JSON; the schedule is written to <dir>/schedule.json
and a final {"done": true|false, ...} line closes the stream.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(var, "2")
os.environ.setdefault("MUJOCO_GL", "egl")

GIB = 1024 ** 3


def apply_limits(soft_gib: float, hard_gib: float, nice: int) -> dict:
    import resource
    applied = {}
    try:
        resource.setrlimit(resource.RLIMIT_DATA, (int(soft_gib * GIB), int(hard_gib * GIB)))
        applied["rlimit_data_gib"] = [soft_gib, hard_gib]
    except (ValueError, OSError) as exc:
        applied["rlimit_data_error"] = str(exc)
    try:
        os.nice(nice)
        applied["nice"] = nice
    except OSError as exc:
        applied["nice_error"] = str(exc)
    applied["threads"] = os.environ["OMP_NUM_THREADS"]
    return applied


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--request", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--soft-gib", type=float, default=4.0)
    ap.add_argument("--hard-gib", type=float, default=5.0)
    ap.add_argument("--nice", type=int, default=5)
    args = ap.parse_args()
    limits = apply_limits(args.soft_gib, args.hard_gib, args.nice)
    print(json.dumps({"progress": 0.0, "label": "worker started", "limits": limits}), flush=True)

    with open(args.request) as handle:
        req = json.load(handle)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        import presentation_compose as pc

        def progress(label: str, frac: float) -> None:
            print(json.dumps({"progress": round(frac, 3), "label": label}), flush=True)

        sched = pc.prepare(req["text"], args.out, voice=req.get("voice", "en-US-AvaNeural"),
                           words_per_minute=float(req.get("words_per_minute", 150.0)),
                           phrase_pause_s=req.get("phrase_pause_s"),
                           target_duration_s=req.get("target_duration_s"),
                           render=bool(req.get("render", True)),
                           progress=progress, name=req.get("name"))

        print(json.dumps({"done": True, "ok": bool(sched["validation"]["ok"]), "duration_s": sched["duration_s"],
                          "gestures": len(sched["gestures"]), "diagnostics": len(sched["diagnostics"])}), flush=True)
        return 0
    except MemoryError:
        print(json.dumps({"done": False, "error": "MemoryError: the worker hit its 4/5 GiB host-RAM limit"}), flush=True)
        return 3
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"done": False, "error": f"{type(exc).__name__}: {exc}",
                          "traceback": traceback.format_exc()[-2000:]}), flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
