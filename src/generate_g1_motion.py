#!/usr/bin/env python3
"""Generate a Unitree G1 motion from a text prompt using cached LLM2Vec embeddings.

Split from the text encoder on purpose: the encoder is an 8B model that needs
most of this machine's memory, while the diffusion model needs 1.21 GB of VRAM.
Running them in separate processes means the encoder's memory is fully released
before Kimodo loads. Encode first with precompute_text_embeddings.py, then run
this.

The cached embedding is supplied through load_model's ``text_encoder=`` parameter,
which Kimodo provides for exactly this purpose, so no Kimodo code is patched.

    python generate_g1_motion.py "raise the right hand and wave" \
        --embeddings outputs/embeddings.npz --output outputs/right_wave
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch


class PrecomputedTextEncoder:
    """Serves cached LLM2Vec embeddings with LLM2VecEncoder's exact call contract.

    Mirrors kimodo.model.LLM2VecEncoder.__call__: a list of prompts yields
    ``(N, 1, 4096)``, a bare string yields ``(1, 4096)``, alongside per-item
    token lengths of 1.
    """

    llm_dim = 4096

    def __init__(self, table: dict[str, np.ndarray], device: str = "cuda") -> None:
        self._table = table
        self._device = device

    def _lookup(self, text: str) -> np.ndarray:
        # Kimodo sanitizes prompts before encoding, so it asks for the sanitized
        # form; accept either spelling so a cache built from raw text still hits.
        from kimodo.sanitize import sanitize_text

        for key in (text, sanitize_text(text)):
            if key in self._table:
                return self._table[key]
        raise KeyError(
            f"No cached embedding for {text!r}. "
            "Re-run precompute_text_embeddings.py with this prompt."
        )

    def __call__(self, text):
        is_string = isinstance(text, str)
        texts = [text] if is_string else list(text)

        stacked = np.stack([self._lookup(t) for t in texts])         # (N, 4096)
        out = torch.tensor(stacked, dtype=torch.float32)[:, None]    # (N, 1, 4096)
        lengths = [1] * len(texts)

        if is_string:
            out, lengths = out[0], lengths[0]
        return out.to(self._device), lengths

    def to(self, device):
        self._device = str(device)
        return self

    def eval(self):
        return self


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt")
    ap.add_argument("--embeddings", default="outputs/embeddings.npz")
    ap.add_argument("--output", required=True, help="Output directory")
    ap.add_argument("--name", default=None, help="Basename (default: derived from output dir)")
    ap.add_argument("--model", default="Kimodo-G1-RP-v1")
    ap.add_argument("--duration", type=float, default=3.0)
    ap.add_argument("--diffusion-steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-clamp", action="store_true",
                    help="Skip apply_g1_real_robot_projection (not recommended)")
    args = ap.parse_args()

    from kimodo.exports.mujoco import MujocoQposConverter, apply_g1_real_robot_projection
    from kimodo.model.load_model import load_model
    from kimodo.skeleton import global_rots_to_local_rots
    from kimodo.tools import seed_everything

    name = args.name or os.path.basename(os.path.normpath(args.output))
    os.makedirs(args.output, exist_ok=True)

    blob = np.load(args.embeddings, allow_pickle=True)
    table = {str(p): e for p, e in zip(blob["prompts"], blob["embeddings"])}
    print(f"loaded {len(table)} cached embedding(s) from {args.embeddings}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = PrecomputedTextEncoder(table, device=device)

    t0 = time.time()
    model, resolved = load_model(
        args.model, device=device, text_encoder=encoder, return_resolved_name=True
    )
    load_s = time.time() - t0
    print(f"model {resolved!r} loaded in {load_s:.1f}s on {device}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    seed_everything(args.seed)

    num_frames = int(round(args.duration * model.fps))
    t0 = time.time()
    out = model(
        [args.prompt],
        num_frames,
        constraint_lst=[],
        num_denoising_steps=args.diffusion_steps,
        num_samples=1,
        multi_prompt=True,
        num_transition_frames=5,
        post_processing=False,   # generate.py disables this for G1
        return_numpy=True,
    )
    gen_s = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    print(f"generated {num_frames} frames in {gen_s:.1f}s (peak VRAM {peak:.2f} GB)")

    npz_path = os.path.join(args.output, f"{name}.npz")
    np.savez(npz_path, **{k: v for k, v in out.items() if hasattr(v, "shape")})

    skeleton = model.skeleton
    posed = torch.from_numpy(out["posed_joints"][0])
    globalr = torch.from_numpy(out["global_rot_mats"][0])

    if args.no_clamp:
        local = torch.from_numpy(out["local_rot_mats"][0])
        root = torch.from_numpy(out["root_positions"][0])
    else:
        # Kimodo output can exceed the G1's mechanical limits; this is the
        # repository's official projection to real-robot DoF with clamping.
        posed, globalr = apply_g1_real_robot_projection(
            skeleton.cpu(), posed, globalr, clamp_to_limits=True
        )
        local = global_rots_to_local_rots(globalr, skeleton.cpu())
        root = posed[:, skeleton.root_idx, :]

    converter = MujocoQposConverter(skeleton.cpu())
    qpos = converter.dict_to_qpos(
        {"local_rot_mats": local[None], "root_positions": root[None]}, torch.device("cpu")
    )
    csv_path = os.path.join(args.output, f"{name}.csv")
    converter.save_csv(qpos, csv_path)

    meta = {
        "prompt": args.prompt,
        "model": resolved,
        "duration_s": args.duration,
        "frames": num_frames,
        "fps": float(model.fps),
        "diffusion_steps": args.diffusion_steps,
        "seed": args.seed,
        "clamped_to_real_robot_limits": not args.no_clamp,
        "generation_s": round(gen_s, 2),
        "peak_vram_gb": round(peak, 3),
        "qpos_shape": list(qpos.shape),
    }
    with open(os.path.join(args.output, f"{name}_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"  npz : {npz_path}")
    print(f"  csv : {csv_path}  shape={tuple(qpos.shape)}")
    print(f"  meta: {os.path.join(args.output, f'{name}_meta.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
