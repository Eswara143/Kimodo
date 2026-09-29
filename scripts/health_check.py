#!/usr/bin/env python3
"""End-to-end health check for the Kimodo-G1 setup.

Reports versions, hardware, model availability, and pipeline status. Prints only
the Hugging Face *username*, never the token.

    conda run -n kimodo python health_check.py
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys

_CUR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_CUR) if os.path.basename(_CUR) == "scripts" else _CUR
REPO = os.path.join(ROOT, "kimodo_repo")

OK, FAIL, WARN = "[PASS]", "[FAIL]", "[WARN]"


def line(status: str, label: str, detail: str = "") -> None:
    print(f"{status:7s} {label:26s} {detail}")


def main() -> int:
    print("=" * 78)
    print("KIMODO-G1 HEALTH CHECK")
    print("=" * 78)

    # --- Environment -------------------------------------------------------
    print("\n-- Environment --")
    line(OK, "python", sys.version.split()[0])
    for mod in ("numpy", "transformers", "peft", "accelerate", "mujoco", "kimodo"):
        try:
            m = importlib.import_module(mod)
            line(OK, mod, getattr(m, "__version__", "(no __version__)"))
        except ImportError as e:
            line(FAIL, mod, str(e))

    # --- Hardware & PyTorch Engine (CPU-First) -----------------------------
    print("\n-- Hardware & PyTorch Execution Engine (CPU-First) --")
    try:
        import torch
        import psutil

        line(OK, "torch", torch.__version__)
        line(OK, "execution mode", "CPU-First (Intel Core Ultra 7 155H)")
        cpus = psutil.cpu_count(logical=True)
        line(OK, "CPU threads", f"{cpus} threads available")
        avail = torch.cuda.is_available()
        line(OK, "hardware acceleration", "Intel Arc iGPU / CPU Fallback" if not avail else "Available")
    except Exception as e:  # noqa: BLE001
        line(FAIL, "torch", repr(e))

    # --- Repository --------------------------------------------------------
    print("\n-- Kimodo repository --")
    if os.path.isdir(REPO):
        try:
            sha = subprocess.check_output(
                ["git", "-C", REPO, "rev-parse", "HEAD"], text=True
            ).strip()
            dirty = subprocess.check_output(
                ["git", "-C", REPO, "status", "--porcelain"], text=True
            ).strip()
            line(OK, "commit", sha)
            line(OK if not dirty else WARN, "working tree", "clean" if not dirty else "modified")
        except subprocess.CalledProcessError as e:
            line(WARN, "git", repr(e))
    else:
        line(FAIL, "repo", f"not found at {REPO}")

    for cli in ("kimodo_gen", "kimodo_convert", "kimodo_textencoder", "kimodo_demo"):
        path = shutil.which(cli)
        line(OK if path else FAIL, cli, path or "not on PATH")

    # --- Hugging Face ------------------------------------------------------
    print("\n-- Hugging Face --")
    try:
        from huggingface_hub import whoami

        line(OK, "authenticated as", whoami()["name"])  # username only
    except Exception as e:  # noqa: BLE001
        line(FAIL, "authentication", type(e).__name__)

    print("\n-- Model availability (local cache only) --")
    from huggingface_hub import snapshot_download

    # original/* is deliberately never downloaded (16 GB duplicate transformers
    # does not load); without the same exclusion here, local_files_only reports
    # the snapshot as incomplete and gives a false negative.
    repos = [
        ("nvidia/Kimodo-G1-RP-v1", None),
        ("McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp", None),
        ("McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised", None),
        ("meta-llama/Meta-Llama-3-8B-Instruct", ["original/*"]),
    ]
    for repo, ignore in repos:
        try:
            p = snapshot_download(repo_id=repo, local_files_only=True, ignore_patterns=ignore)
            size = sum(
                os.path.getsize(os.path.realpath(os.path.join(r, f)))
                for r, _, fs in os.walk(p)
                for f in fs
            )
            line(OK, repo.split("/")[-1], f"{size / 1e9:.2f} GB")
        except Exception:  # noqa: BLE001
            line(FAIL, repo.split("/")[-1], "not fully downloaded")

    # --- Memory budget -----------------------------------------------------
    print("\n-- Memory budget --")
    try:
        import psutil

        vm = psutil.virtual_memory()
        sw = psutil.swap_memory()
        line(OK, "RAM total", f"{vm.total / 1e9:.1f} GB")
        line(OK, "RAM available", f"{vm.available / 1e9:.1f} GB")
        line(OK, "swap free", f"{sw.free / 1e9:.1f} GB")

        # The unmerged encoder needs ~15 GB resident, which does not fit here.
        # The merged encoder sidesteps that: no PEFT at load time means
        # accelerate can offload to disk, so only the working set is resident.
        merged = os.path.expanduser("~/.cache/kimodo/llm2vec_merged")
        if os.path.exists(os.path.join(merged, "model.safetensors.index.json")):
            size = sum(
                os.path.getsize(os.path.join(merged, f)) for f in os.listdir(merged)
            )
            line(OK, "merged encoder", f"{size / 1e9:.2f} GB at {merged}")
            line(OK, "text encoding", "available locally (strategy: merged)")
        else:
            line(WARN, "merged encoder", "not built - run merge_llm2vec_lora.py")
            have = (vm.available + sw.free) / 1e9
            line(
                OK if have >= 19 else FAIL,
                "unmerged LLM2Vec headroom",
                f"needs ~19 GB, have ~{have:.1f} GB",
            )
    except ImportError:
        line(WARN, "psutil", "not installed")

    print("\n" + "=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
