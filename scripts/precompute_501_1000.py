#!/usr/bin/env python3
"""Precompute and Cache LLM2Vec Text Embeddings for Gestures 501–1000.

Loads all 500 motion prompts from gesture_library_501_1000.json, encodes them
using Kimodo's pre-merged LLM2Vec text encoder in memory-conscious batches,
and saves outputs/embeddings_501_1000.npz.
"""

from __future__ import annotations

import json
import os
import sys
import time
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import precompute_text_embeddings as pte
from kimodo.sanitize import sanitize_text

JSON_501_1000 = os.path.join(ROOT, "gesture_library_501_1000.json")
OUT_EMBEDDINGS = os.path.join(ROOT, "outputs", "embeddings_501_1000.npz")
MERGED_DIR = os.path.expanduser("~/.cache/kimodo/llm2vec_merged")


def precompute_all_501_1000():
    if not os.path.exists(JSON_501_1000):
        raise FileNotFoundError(f"Missing {JSON_501_1000}. Run extract_pdf_specifications.py first.")

    with open(JSON_501_1000, "r") as f:
        data = json.load(f)

    gestures = data.get("gestures", [])
    raw_prompts = [g["motion_prompt"] for g in gestures]
    sanitized_prompts = [sanitize_text(p) for p in raw_prompts]
    
    print(f"Loaded {len(sanitized_prompts)} motion prompts from {JSON_501_1000}")
    t0 = time.time()

    gpu_budget = 0.0
    print(f"Loading pre-merged LLM2Vec encoder (GPU VRAM budget: {gpu_budget} GiB)...")
    l2v = pte.load_merged(gpu_gib=gpu_budget, cpu_gib=12.0, offload_dir="/tmp/kimodo_llm2vec_offload", merged_dir=MERGED_DIR)

    # Encode in batches of 25 prompts to maintain low RAM and steady progress
    batch_size = 25
    all_embeddings = []
    
    for i in range(0, len(sanitized_prompts), batch_size):
        batch = sanitized_prompts[i : i + batch_size]
        print(f"Encoding prompts {i+1}..{min(i+batch_size, len(sanitized_prompts))}/{len(sanitized_prompts)}...")
        emb = pte.encode_sharded(l2v, batch)
        all_embeddings.append(emb)

    embeddings = np.concatenate(all_embeddings, axis=0)
    dt = time.time() - t0

    print(f"\nCompleted encoding in {dt:.1f}s. Shape: {embeddings.shape}")
    os.makedirs(os.path.dirname(OUT_EMBEDDINGS), exist_ok=True)
    
    # Save with both raw and sanitized prompt keys for seamless lookup
    np.savez(OUT_EMBEDDINGS, prompts=np.array(sanitized_prompts, dtype=object), embeddings=embeddings)
    print(f"Saved {len(embeddings)} pre-computed embeddings to {OUT_EMBEDDINGS}")
    
    # Release text encoder memory explicitly
    del l2v
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    precompute_all_501_1000()
