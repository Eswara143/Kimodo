#!/usr/bin/env python3
"""Encode prompts with Kimodo's LLM2Vec text encoder and cache the embeddings.

Why this exists: Kimodo's text encoder is Meta-Llama-3-8B, ~16.06 GB in bfloat16.
That does not fit in this laptop's ~11 GB of free RAM, but it does fit across
6 GB of VRAM plus system RAM if accelerate is allowed to shard it. The diffusion
model only needs 1.21 GB of VRAM and runs in a *separate* later process, so the
encoder is free to use the whole GPU while it runs.

Two strategies, tried in order:

  cpu       the stock Kimodo path (LLM2VecEncoder with TEXT_ENCODER_DEVICE=cpu),
            supervised by a watchdog that aborts before the machine starts
            thrashing rather than inviting the OOM killer.
  sharded   accelerate device_map="auto" across GPU + CPU (+ disk if needed).
            encode() normally calls .to(device), which would undo accelerate's
            placement, so that one call is neutralised; every other part of the
            official encode path -- instruction formatting, pooling, dtype -- is
            left exactly as upstream wrote it.

Embeddings are consumed by generate_g1_motion.py through load_model's supported
``text_encoder=`` injection point, so Kimodo itself is never modified.

    python precompute_text_embeddings.py --out outputs/embeddings.npz \
        "raise the right hand and wave" "raise the left hand and wave"
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time

import numpy as np
import psutil
import torch

BASE = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"
PEFT = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised"
LLM_DIM = 4096


def _gb(x: float) -> str:
    return f"{x / 1e9:.2f} GB"


class MemoryWatchdog:
    """Abort the process if RSS crosses a ceiling, instead of hitting the OOM killer."""

    def __init__(self, ceiling_bytes: int) -> None:
        self.ceiling = ceiling_bytes
        self.peak = 0
        self.tripped = False
        self._stop = threading.Event()
        self._proc = psutil.Process()

    def _run(self) -> None:
        while not self._stop.wait(0.25):
            rss = self._proc.memory_info().rss
            self.peak = max(self.peak, rss)
            if rss > self.ceiling:
                self.tripped = True
                print(
                    f"\n  watchdog: RSS {_gb(rss)} exceeded ceiling {_gb(self.ceiling)}; aborting",
                    flush=True,
                )
                os._exit(42)  # hard exit: unwinding a half-loaded 8B model is not useful

    def __enter__(self) -> "MemoryWatchdog":
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()


def load_cpu():
    """Stock Kimodo path: LLM2VecEncoder pinned to CPU."""
    from kimodo.model import LLM2VecEncoder

    return LLM2VecEncoder(
        base_model_name_or_path=BASE,
        peft_model_name_or_path=PEFT,
        dtype="bfloat16",
        llm_dim=LLM_DIM,
        device="cpu",
    )


def load_sharded(gpu_gib: float, cpu_gib: float, offload_dir: str):
    """accelerate-sharded load across GPU + CPU (+ disk), bypassing the .to() move."""
    from kimodo.model.llm2vec.llm2vec import LLM2Vec

    max_memory = {"cpu": f"{cpu_gib:g}GiB"}
    if torch.cuda.is_available():
        max_memory[0] = f"{gpu_gib:g}GiB"
    os.makedirs(offload_dir, exist_ok=True)
    print(f"  device_map=auto  max_memory={max_memory}  offload={offload_dir}", flush=True)

    l2v = LLM2Vec.from_pretrained(
        base_model_name_or_path=BASE,
        peft_model_name_or_path=PEFT,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
        offload_folder=offload_dir,
        low_cpu_mem_usage=True,
    )
    # accelerate already placed every submodule; encode()/_encode() would call
    # .to(device) and collapse that placement onto one device.
    l2v.to = lambda *a, **k: l2v  # type: ignore[method-assign]
    return l2v


def load_sharded_nomerge(gpu_gib: float, cpu_gib: float, offload_dir: str):
    """Sharded load that never calls merge_and_unload().

    PEFT refuses to merge adapters into disk-offloaded modules, and with a 16.06 GB
    model against ~14 GB of GPU+RAM some disk offload is unavoidable. Leaving both
    LoRAs unmerged is mathematically equivalent -- merging only folds the delta into
    the base weight ahead of time instead of adding it during the forward pass --
    and it keeps peak memory at base + adapters.

    Mirrors LLM2Vec.from_pretrained: bidirectional Llama, tokenizer with left
    padding and eos as pad, supervised LoRA on top, LLM2Vec's default mean pooling
    (neither adapter repo ships an llm2vec_config.json, so upstream would use the
    same defaults).
    """
    from peft import PeftModel
    from transformers import AutoTokenizer

    from kimodo.model.llm2vec.llm2vec import LLM2Vec
    from kimodo.model.llm2vec.models.bidirectional_llama import LlamaBiModel

    max_memory = {"cpu": f"{cpu_gib:g}GiB"}
    if torch.cuda.is_available():
        max_memory[0] = f"{gpu_gib:g}GiB"
    os.makedirs(offload_dir, exist_ok=True)
    print(f"  device_map=auto  max_memory={max_memory}  offload={offload_dir}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # BASE carries adapter_config.json, so transformers pulls the Llama-3 base
    # weights it names and attaches the MNTP LoRA during this call.
    model = LlamaBiModel.from_pretrained(
        BASE,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
        offload_folder=offload_dir,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(model, PEFT)  # supervised LoRA, unmerged
    model.eval()

    l2v = LLM2Vec(model=model, tokenizer=tokenizer)
    l2v.to = lambda *a, **k: l2v  # type: ignore[method-assign]
    return l2v


def load_merged(gpu_gib: float, cpu_gib: float, offload_dir: str, merged_dir: str):
    """Load pre-merged encoder weights: no PEFT, so disk offload is safe.

    merge_llm2vec_lora.py folds both LoRAs into the base weights ahead of time.
    That removes the reason the sharded strategies fail here -- PEFT cannot apply
    an adapter to a disk-offloaded module -- so accelerate is free to place part
    of the model on disk and the whole 15.01 GB encoder fits on a 16 GB machine.
    """
    from kimodo.model.llm2vec.llm2vec import LLM2Vec
    from kimodo.model.llm2vec.models.bidirectional_llama import LlamaBiModel
    from transformers import AutoTokenizer

    if not os.path.exists(os.path.join(merged_dir, "model.safetensors.index.json")):
        raise FileNotFoundError(
            f"No merged encoder at {merged_dir}. Run: python merge_llm2vec_lora.py"
        )

    max_memory = {"cpu": f"{cpu_gib:g}GiB"}
    if torch.cuda.is_available():
        max_memory[0] = f"{gpu_gib:g}GiB"
    os.makedirs(offload_dir, exist_ok=True)
    print(f"  merged_dir={merged_dir}", flush=True)
    print(f"  device_map=auto  max_memory={max_memory}  offload={offload_dir}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(merged_dir)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = LlamaBiModel.from_pretrained(
        merged_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
        offload_folder=offload_dir,
        low_cpu_mem_usage=True,
    )
    model.eval()
    # prepare_for_tokenization only applies the Llama-3 chat template when this
    # is exactly the upstream name; loading from a local dir would otherwise
    # overwrite it with the path and silently change the embeddings.
    model.config._name_or_path = "meta-llama/Meta-Llama-3-8B-Instruct"

    l2v = LLM2Vec(model=model, tokenizer=tokenizer)
    l2v.to = lambda *a, **k: l2v  # type: ignore[method-assign]
    return l2v


def encode_sharded(l2v, prompts: list[str]) -> np.ndarray:
    """Run the official encode() path on a sharded model."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    emb = l2v.encode(prompts, batch_size=1, show_progress_bar=False, device=dev)
    return np.asarray(emb.float().cpu())


def encode_official(encoder, prompts: list[str]) -> np.ndarray:
    """LLM2VecEncoder.__call__ returns (T,1,4096); collapse to (T,4096)."""
    out, _ = encoder(prompts)
    return np.asarray(out.float().cpu())[:, 0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompts", nargs="+", help="Prompts to encode")
    ap.add_argument("--out", default="outputs/embeddings.npz")
    ap.add_argument(
        "--strategy",
        choices=("auto", "cpu", "sharded", "sharded-nomerge", "merged"),
        default="auto",
        help="sharded-nomerge must be run as its own process: a failed 'sharded' "
             "attempt leaves several GiB pinned on the GPU that cannot be reclaimed "
             "in-process, which then OOMs the retry.",
    )
    ap.add_argument("--gpu-gib", type=float, default=4.5, help="VRAM budget for the encoder")
    ap.add_argument("--cpu-gib", type=float, default=9.0, help="System RAM budget for the encoder")
    ap.add_argument("--offload-dir", default="/tmp/kimodo_llm2vec_offload")
    ap.add_argument("--merged-dir",
                    default=os.path.expanduser("~/.cache/kimodo/llm2vec_merged"),
                    help="Pre-merged encoder from merge_llm2vec_lora.py")
    args = ap.parse_args()

    vm = psutil.virtual_memory()
    print("=" * 70)
    print("LLM2VEC TEXT EMBEDDING PRECOMPUTE")
    print("=" * 70)
    print(f"prompts        : {len(args.prompts)}")
    print(f"RAM available  : {_gb(vm.available)}")
    if torch.cuda.is_available():
        print(f"GPU            : {torch.cuda.get_device_name(0)} "
              f"{_gb(torch.cuda.get_device_properties(0).total_memory)}")
    print(f"bf16 model need: ~16.06 GB")
    print()

    # Kimodo sanitizes prompts (capitalisation, final punctuation) before handing
    # them to the text encoder, so encode and key on the sanitized form or the
    # cache lookup at generation time will miss.
    from kimodo.sanitize import sanitize_text

    prompts = [sanitize_text(p) for p in args.prompts]
    for raw, clean in zip(args.prompts, prompts):
        if raw != clean:
            print(f"sanitized: {raw!r} -> {clean!r}")
    args.prompts = prompts

    strategy = args.strategy
    if strategy == "auto":
        # 16.06 GB of weights plus headroom for activations and the shard being read.
        merged_ready = os.path.exists(
            os.path.join(os.path.expanduser("~/.cache/kimodo/llm2vec_merged"),
                         "model.safetensors.index.json"))
        if merged_ready:
            strategy = "merged"
        else:
            strategy = "cpu" if vm.available > 19e9 else "sharded"
        print(f"auto-selected strategy: {strategy}")
        print()

    t0 = time.time()
    embeddings = None

    if strategy == "cpu":
        ceiling = int(vm.available * 0.92)
        print(f"[cpu] loading, watchdog ceiling {_gb(ceiling)}", flush=True)
        with MemoryWatchdog(ceiling) as wd:
            enc = load_cpu()
            print(f"[cpu] loaded, peak RSS {_gb(wd.peak)}", flush=True)
            embeddings = encode_official(enc, args.prompts)
    else:
        if strategy == "sharded":
            # Single attempt. On failure the caller must re-invoke with
            # --strategy sharded-nomerge in a *new* process: a half-loaded 8B
            # model cannot be evicted from the GPU within this one.
            print("[sharded] loading (stock loader, includes merge_and_unload)", flush=True)
            try:
                l2v = load_sharded(args.gpu_gib, args.cpu_gib, args.offload_dir)
            except Exception as e:  # noqa: BLE001
                print(f"[sharded] failed: {type(e).__name__}: {str(e)[:300]}")
                print("[sharded] re-run with --strategy sharded-nomerge (fresh process)")
                return 2
        elif strategy == "merged":
            print("[merged] loading pre-merged encoder (no PEFT in the load path)", flush=True)
            l2v = load_merged(args.gpu_gib, args.cpu_gib, args.offload_dir, args.merged_dir)
        else:
            print("[sharded-nomerge] loading", flush=True)
            l2v = load_sharded_nomerge(args.gpu_gib, args.cpu_gib, args.offload_dir)

        rss = psutil.Process().memory_info().rss
        print(f"[{strategy}] loaded  host RSS {_gb(rss)}"
              + (f"  VRAM {_gb(torch.cuda.memory_allocated())}" if torch.cuda.is_available() else ""),
              flush=True)
        embeddings = encode_sharded(l2v, args.prompts)

    dt = time.time() - t0
    if embeddings.ndim != 2 or embeddings.shape[-1] != LLM_DIM:
        print(f"ERROR: expected (N,{LLM_DIM}) embeddings, got {embeddings.shape}")
        return 1
    if not np.all(np.isfinite(embeddings)):
        print("ERROR: embeddings contain non-finite values")
        return 1

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, prompts=np.array(args.prompts, dtype=object), embeddings=embeddings)

    print()
    print(f"strategy   : {strategy}")
    print(f"time       : {dt:.1f}s")
    print(f"embeddings : {embeddings.shape} {embeddings.dtype}")
    print(f"norms      : {np.linalg.norm(embeddings, axis=1)}")
    print(f"saved      : {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
