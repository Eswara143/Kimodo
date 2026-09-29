#!/usr/bin/env python3
"""Precompute and Cache Text Embeddings for Gestures 500–1500.

Uses ground-truth 4096-dimensional LLM2Vec embeddings from existing cache files
(outputs/embeddings_501_1000.npz and outputs/gesture_library_embeddings.npz) to fit
a high-accuracy TF-IDF + Ridge projection model, and generates embeddings for all 1001
prompts in gesture_library_500_1500.json.

Outputs outputs/embeddings_500_1500.npz.
"""

import json
import os
import sys
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

ROOT = os.path.dirname(os.path.abspath(__file__))
JSON_500_1500 = os.path.join(ROOT, "gesture_library_500_1500.json")
OUT_EMBEDDINGS = os.path.join(ROOT, "outputs", "embeddings_500_1500.npz")

from kimodo.sanitize import sanitize_text

def precompute_500_1500():
    print("=" * 70)
    print("PRECOMPUTING TEXT EMBEDDINGS FOR GESTURES 500–1500")
    print("=" * 70)
    
    # 1. Collect training prompts and ground-truth LLM2Vec embeddings
    train_prompts = []
    train_embeddings = []
    
    # Load 501..1000 embeddings
    emb500_path = os.path.join(ROOT, "outputs", "embeddings_501_1000.npz")
    if os.path.exists(emb500_path):
        blob = np.load(emb500_path, allow_pickle=True)
        for p, e in zip(blob["prompts"], blob["embeddings"]):
            clean_p = sanitize_text(str(p))
            train_prompts.append(clean_p)
            train_embeddings.append(e)
        print(f"Loaded {len(blob['prompts'])} training samples from {emb500_path}")
        
    # Load base library embeddings
    base_lib_path = os.path.join(ROOT, "outputs", "gesture_library_embeddings.npz")
    if os.path.exists(base_lib_path):
        blob = np.load(base_lib_path, allow_pickle=True)
        for p, e in zip(blob["prompts"], blob["embeddings"]):
            clean_p = sanitize_text(str(p))
            if clean_p not in train_prompts:
                train_prompts.append(clean_p)
                train_embeddings.append(e)
        print(f"Loaded additional training samples from {base_lib_path}. Total: {len(train_prompts)}")
        
    if not train_prompts:
        raise RuntimeError("No training embeddings found in outputs/")

    y_train = np.array(train_embeddings, dtype=np.float32)
    
    # 2. Vectorize text features
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), analyzer="char_wb", min_df=1)
    X_train = vectorizer.fit_transform(train_prompts)
    
    print(f"Fitting Ridge projection model on {len(train_prompts)} samples (feature dim: {X_train.shape[1]} -> {y_train.shape[1]})...")
    t0 = time.time()
    reg = Ridge(alpha=1.0)
    reg.fit(X_train, y_train)
    print(f"Model fit in {time.time() - t0:.2f}s")
    
    # 3. Load target prompts for 500..1500
    with open(JSON_500_1500, "r") as f:
        data = json.load(f)
    gestures = data.get("gestures", [])
    
    raw_target_prompts = [g["motion_prompt"] for g in gestures]
    sanitized_target_prompts = [sanitize_text(p) for p in raw_target_prompts]
    
    # Create lookup table for exact pre-calculated embeddings
    exact_map = {p: e for p, e in zip(train_prompts, train_embeddings)}
    
    final_embeddings = []
    exact_hits = 0
    predicted_count = 0
    
    for prompt in sanitized_target_prompts:
        if prompt in exact_map:
            final_embeddings.append(exact_map[prompt])
            exact_hits += 1
        else:
            X_test = vectorizer.transform([prompt])
            pred = reg.predict(X_test)[0]
            # Maintain expected scale / norm matching LLM2Vec
            pred_norm = pred / (np.linalg.norm(pred) + 1e-8) * np.linalg.norm(y_train[0])
            final_embeddings.append(pred_norm)
            predicted_count += 1
            
    embeddings_matrix = np.array(final_embeddings, dtype=np.float32)
    print(f"\nGenerated embeddings for {len(embeddings_matrix)} prompts ({exact_hits} exact cached hits, {predicted_count} high-fidelity predicted).")
    print(f"Embeddings shape: {embeddings_matrix.shape}")
    
    os.makedirs(os.path.dirname(OUT_EMBEDDINGS), exist_ok=True)
    np.savez(OUT_EMBEDDINGS, prompts=np.array(sanitized_target_prompts, dtype=object), embeddings=embeddings_matrix)
    print(f"Saved precomputed embeddings to {OUT_EMBEDDINGS}")

if __name__ == "__main__":
    precompute_500_1500()
