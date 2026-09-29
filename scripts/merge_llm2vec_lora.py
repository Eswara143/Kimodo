#!/usr/bin/env python3
"""Fold both LLM2Vec LoRA adapters into the Llama-3 base weights, shard by shard.

Why: on a 16 GB machine the 15.01 GB encoder needs accelerate to offload part of
itself to disk, and PEFT cannot apply a LoRA to a disk-offloaded module -- it
raises KeyError on '<layer>.base_layer.weight' both when merging
(merge_and_unload) and when stacking the second adapter
(PeftModel.from_pretrained). Merging offline removes PEFT from the load path
entirely, after which disk offload is perfectly safe.

Peak memory is one shard (~5 GB) plus both adapters (~0.34 GB), instead of the
whole 15 GB model.

Merge maths is the standard LoRA identity, applied in adapter order (MNTP was
trained first, the supervised adapter on top of it):

    W' = W + (alpha / r) * (B @ A)

Accumulation is done in float32 and cast back to bfloat16 only at the end, so
the two sequential updates do not compound bf16 rounding error.

    python merge_llm2vec_lora.py --out ~/.cache/kimodo/llm2vec_merged
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

BASE_REPO = "meta-llama/Meta-Llama-3-8B-Instruct"
MNTP_REPO = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"
SUP_REPO = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised"

# LLM2Vec.prepare_for_tokenization applies the Llama-3 chat template only when
# config._name_or_path is exactly this. The MNTP config declares it, so the
# template is intended; preserve it or the embeddings differ.
CANONICAL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"


def load_adapter(repo: str) -> tuple[dict[str, torch.Tensor], float]:
    """Return {module_name: (A, B)} keyed by base-model module, and the scale."""
    path = snapshot_download(repo, local_files_only=True)
    cfg = json.load(open(os.path.join(path, "adapter_config.json")))
    scale = cfg["lora_alpha"] / cfg["r"]

    pairs: dict[str, dict[str, torch.Tensor]] = {}
    with safe_open(os.path.join(path, "adapter_model.safetensors"), framework="pt") as f:
        for key in f.keys():
            # base_model.model.layers.0.mlp.down_proj.lora_A.weight
            #   -> module "model.layers.0.mlp.down_proj", slot "A"
            stripped = key.removeprefix("base_model.model.")
            for slot in ("lora_A", "lora_B"):
                suffix = f".{slot}.weight"
                if stripped.endswith(suffix):
                    module = "model." + stripped[: -len(suffix)]
                    pairs.setdefault(module, {})[slot[-1]] = f.get_tensor(key)
                    break

    merged = {}
    for module, ab in pairs.items():
        if "A" not in ab or "B" not in ab:
            raise ValueError(f"{repo}: {module} is missing a LoRA A/B half")
        merged[module] = (ab["A"], ab["B"])
    return merged, scale


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.expanduser("~/.cache/kimodo/llm2vec_merged"))
    ap.add_argument("--force", action="store_true", help="Rebuild even if --out already looks complete")
    args = ap.parse_args()

    out = os.path.abspath(args.out)
    done_marker = os.path.join(out, "model.safetensors.index.json")
    if os.path.exists(done_marker) and not args.force:
        print(f"Merged encoder already present at {out} (use --force to rebuild)")
        return 0

    base = snapshot_download(BASE_REPO, local_files_only=True, ignore_patterns=["original/*"])
    mntp, mntp_scale = load_adapter(MNTP_REPO)
    sup, sup_scale = load_adapter(SUP_REPO)
    print(f"MNTP adapter      : {len(mntp)} modules, scale {mntp_scale}")
    print(f"supervised adapter: {len(sup)} modules, scale {sup_scale}")

    os.makedirs(out, exist_ok=True)
    index = json.load(open(os.path.join(base, "model.safetensors.index.json")))
    shards = sorted(set(index["weight_map"].values()))

    new_weight_map: dict[str, str] = {}
    total_size = 0
    merged_count = 0

    for shard in shards:
        src = os.path.join(base, shard)
        tensors: dict[str, torch.Tensor] = {}
        print(f"\n{shard}", flush=True)

        with safe_open(src, framework="pt") as f:
            for key in f.keys():
                if key == "lm_head.weight":
                    # LlamaBiModel is an encoder body and never uses lm_head;
                    # dropping it saves 1.05 GB of disk and load time.
                    continue

                w = f.get_tensor(key)
                module = key.removesuffix(".weight")

                if module in mntp or module in sup:
                    acc = w.to(torch.float32)
                    for table, scale in ((mntp, mntp_scale), (sup, sup_scale)):
                        if module in table:
                            a, b = table[module]
                            acc += scale * (b.to(torch.float32) @ a.to(torch.float32))
                    w = acc.to(torch.bfloat16)
                    merged_count += 1
                    del acc

                tensors[key] = w
                new_weight_map[key] = shard
                total_size += w.numel() * w.element_size()

        dst = os.path.join(out, shard)
        save_file(tensors, dst, metadata={"format": "pt"})
        print(f"  wrote {dst}  ({os.path.getsize(dst) / 1e9:.2f} GB)", flush=True)
        del tensors

    json.dump(
        {"metadata": {"total_size": total_size}, "weight_map": new_weight_map},
        open(os.path.join(out, "model.safetensors.index.json"), "w"),
        indent=2,
    )

    # Config comes from the base model so it is a plain LlamaConfig with no
    # adapter_config.json beside it -- that absence is the point, it keeps
    # transformers from re-entering the PEFT path at load time.
    cfg = json.load(open(os.path.join(base, "config.json")))
    cfg["_name_or_path"] = CANONICAL_NAME
    cfg["torch_dtype"] = "bfloat16"
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2)

    for fname in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        src = os.path.join(base, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(out, fname))

    print(f"\nmerged {merged_count} weight tensors "
          f"(expected {len(set(mntp) | set(sup))})")
    print(f"merged encoder written to {out}  ({total_size / 1e9:.2f} GB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
