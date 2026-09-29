#!/usr/bin/env python3
"""Batch generate all 250 Unitree G1 gesture motions via webapp API.

Idempotent: skips gestures that are already generated and valid.
Retries with alternative seeds if structural validation fails.
"""

import json
import os
import sys
import time
import urllib.request
import urllib.error

ROOT = os.path.dirname(os.path.abspath(__file__))
GESTURE_JSON = os.path.join(ROOT, "gesture_library_250.json")
OUT_DIR = os.path.join(ROOT, "outputs", "web", "pdf_250_library")
API_URL = "http://127.0.0.1:8000/api/generate"
STATUS_URL = "http://127.0.0.1:8000/api/status"


def check_api() -> bool:
    try:
        with urllib.request.urlopen(STATUS_URL, timeout=5) as resp:
            data = json.load(resp)
            return data.get("ready", False)
    except Exception as e:
        print(f"Error checking API status: {e}")
        return False


def generate_gesture(g: dict, seed: int = 42, steps: int = 30) -> dict | None:
    gid = g["gesture_id"]
    prompt = g["kimodo_prompt"]
    payload = json.dumps({
        "prompt": prompt,
        "duration": 3.0,
        "steps": steps,
        "seed": seed,
        "clamp": True,
        "name": gid,
        "collection": "pdf_250_library"
    }).encode("utf-8")

    req = urllib.request.Request(
        API_URL,
        data=payload,
        headers={"Content-Type": "application/json"}
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:
        print(f"HTTP error for {gid}: {e.code} - {e.read().decode('utf-8')[:100]}")
    except Exception as e:
        print(f"Exception for {gid}: {e}")
    return None


def main():
    if not os.path.exists(GESTURE_JSON):
        print(f"Error: {GESTURE_JSON} not found!")
        sys.exit(1)

    with open(GESTURE_JSON, "r") as f:
        data = json.load(f)

    gestures = data.get("gestures", [])
    total = len(gestures)
    print(f"Loaded {total} gestures from {GESTURE_JSON}.")

    if not check_api():
        print("Waiting for backend API to be ready...")
        for _ in range(30):
            if check_api():
                break
            time.sleep(2)
        if not check_api():
            print("Error: Backend API not reachable/ready at http://127.0.0.1:8000")
            sys.exit(1)

    print("Backend API is ready! Starting batch generation...\n")

    os.makedirs(OUT_DIR, exist_ok=True)

    passed = 0
    failed = 0
    skipped = 0

    t_start = time.time()

    for idx, g in enumerate(gestures, 1):
        gid = g["gesture_id"]
        number = g["number"]
        intent = g["intent"]
        cat = g["category"]

        meta_path = os.path.join(OUT_DIR, gid, "meta.json")
        mp4_path = os.path.join(OUT_DIR, gid, f"{gid}.mp4")

        # Check if already generated and valid
        if os.path.isfile(meta_path) and os.path.isfile(mp4_path):
            try:
                meta = json.load(open(meta_path))
                if meta.get("valid", False):
                    print(f"[{idx}/{total}] {gid} ({number}: {intent}) — Already generated & valid (Skipping)")
                    passed += 1
                    skipped += 1
                    continue
            except Exception:
                pass

        print(f"[{idx}/{total}] Generating {gid} (#{number} '{intent}' in {cat})...", flush=True)

        seeds = [42, 101, 202, 303, 505]
        success = False

        for s in seeds:
            res = generate_gesture(g, seed=s, steps=30)
            if res and res.get("valid", False):
                gen_s = res.get("generation_s", 0)
                print(f"   ✓ OK ({gen_s:.1f}s) seed={s}")
                success = True
                passed += 1
                break
            else:
                print(f"   ✗ Seed {s} failed or invalid, retrying...")

        if not success:
            print(f"   ✖ FAILED to generate valid motion for {gid} ({intent}) after retries.")
            failed += 1

    t_total = time.time() - t_start
    print("\n==================================================")
    print(f"BATCH GENERATION COMPLETE in {t_total / 60:.1f} minutes!")
    print(f"Total: {total} | Passed: {passed} | Skipped: {skipped} | Failed: {failed}")
    print("==================================================")


if __name__ == "__main__":
    main()
