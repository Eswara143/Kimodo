"""Download the model assets Kimodo needs for G1 generation.

Excludes meta-llama's original/*.pth consolidated checkpoint (~16 GB): transformers
loads the sharded .safetensors instead, so that copy is dead weight on disk.
snapshot_download resumes partial blobs automatically.
"""

from huggingface_hub import snapshot_download

REPOS = [
    # Kimodo G1 diffusion checkpoint.
    ("nvidia/Kimodo-G1-RP-v1", None),
    # LLM2Vec text encoder: the MNTP LoRA is the base_model_name_or_path Kimodo
    # asks for, the -supervised LoRA is layered on top, and both resolve their
    # own base weights to meta-llama/Meta-Llama-3-8B-Instruct below.
    ("McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp", None),
    ("McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised", None),
    ("meta-llama/Meta-Llama-3-8B-Instruct", ["original/*"]),
]

for repo_id, ignore in REPOS:
    print(f"=== Downloading {repo_id} ===", flush=True)
    path = snapshot_download(repo_id=repo_id, ignore_patterns=ignore)
    print(f"=== Done: {repo_id} -> {path} ===", flush=True)

print("ALL DOWNLOADS COMPLETE", flush=True)
