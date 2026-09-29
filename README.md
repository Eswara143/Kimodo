# NVIDIA Kimodo → Unitree G1 (local setup)

Text prompt → Kimodo-G1 motion → MuJoCo qpos CSV → offline validation.

Everything here is **simulation-side only**. No part of this setup opens a
connection to a physical robot, and the validation step is explicitly not
hardware authorization. See [Safety](#safety).

## Machine this was built on

| | |
|---|---|
| OS | Ubuntu 22.04 |
| GPU | NVIDIA GeForce RTX 3050 6GB Laptop GPU |
| Driver | 580.173.02 (max CUDA 13.0) |
| RAM | 16 GB |
| Conda | miniforge3 |

**System RAM, not VRAM, is the binding constraint on this machine.** The
CPU-text-encoder configuration below solves the 6 GB VRAM problem — it drops GPU
usage from ~17 GB to <3 GB — but it moves the 8B LLM2Vec encoder into system
RAM, where bfloat16 weights need ~16.06 GB. This machine has 15 GiB total
(~10 GiB free) plus a 2 GiB swapfile. LLM2Vec's `merge_and_unload()` materializes
the LoRA-targeted projections (~13.96 GB of the 16.06 GB) into resident memory
rather than leaving them memory-mapped, so the load does not fit.

### Local text encoding: solved by merging the LoRAs offline

**Status: working locally.** Run this once, then everything below works on the
laptop:

```bash
python merge_llm2vec_lora.py     # ~15 GB written to ~/.cache/kimodo/llm2vec_merged
```

`precompute_text_embeddings.py` auto-detects the merged encoder and uses it.
Measured: load + encode in **11.1 s**, host RSS **5.46 GB**, VRAM **4.54 GB**.

Why this was needed, and why it works:

### The blocker it removes

This was measured, not predicted. `LlamaBiModel` (the bidirectional encoder body,
which drops `lm_head`) is **15.01 GB** in bf16. Against ~5.5 GB usable VRAM plus
~8 GB free RAM — about 13.5 GB — it is short by roughly 1.5 GB, so accelerate
must offload the remainder to disk. **Disk offload is incompatible with PEFT
here**, and both LoRAs go through PEFT:

| attempt | failure |
|---|---|
| `device_map="auto"` + `merge_and_unload()` | `KeyError: 'layers.14.self_attn.q_proj.base_layer.weight'` |
| same, skipping the merge | `PeftModel.from_pretrained` → `KeyError: 'layers.17...base_layer.weight'` building `offload_index` |

Also worth knowing: a failed attempt leaves several GiB pinned on the GPU that
cannot be reclaimed in-process, so a retry OOMs. Strategies must each run in a
fresh process — which is why `sharded-nomerge` is an explicit `--strategy` rather
than an automatic in-process fallback.

The fix: fold both LoRAs into the base weights *before* loading, one shard at a
time (`merge_llm2vec_lora.py`). Peak memory during the merge is one shard
(~5 GB) plus both adapters (~0.34 GB). The result is a plain model with no
`adapter_config.json` beside it, so transformers never enters the PEFT path and
accelerate is free to offload to disk.

Merge details worth knowing if you touch that script:

- `W' = W + (alpha/r) * (B @ A)`, scale 2.0 for both adapters, applied in order
  (MNTP first, supervised on top — that is how they were trained). Accumulated
  in float32 and cast back to bf16 once, so two sequential updates do not
  compound rounding error.
- 224 modules per adapter (32 layers × 7 projections). The script asserts it
  merged exactly that many.
- `lm_head` is dropped — `LlamaBiModel` is an encoder body and never uses it,
  saving 1.05 GB.
- `config._name_or_path` is forced back to `meta-llama/Meta-Llama-3-8B-Instruct`.
  `LLM2Vec.prepare_for_tokenization` applies the Llama-3 chat template *only*
  when that string matches exactly; loading from a local directory would
  otherwise overwrite it with the path and silently change every embedding.

`kaggle_kimodo_g1.py` remains available and self-contained if you would rather
run on Kaggle's ~30 GB, where the stock CPU path loads without any of this.

## Repository Layout

```text
kimodo/
├── src/                      # Core application, web services, adapters, pipelines, and modules
│   ├── webapp.py             # FastAPI backend API & supervisor target
│   ├── webui.html            # Unified frontend interface
│   ├── presentation_compose.py # Speech + gesture timeline composition
│   ├── presentation_safety.py  # MuJoCo forward kinematics allowlist filter
│   ├── presentation_worker.py  # Presentation rendering worker process
│   ├── compose_sentence_gestures.py # Sentence composition pipeline
│   ├── video_gesture_pipeline.py # 2D/3D video-to-gesture extraction & retargeting
│   ├── g1_config.py          # Dynamic Unitree G1 robot configuration & joint limits
│   ├── g1_robot_adapter.py   # Robot control adapter (simulation & preview modes)
│   ├── convert_video_to_csv_and_gesture.py # Video-to-motion CLI conversion
│   ├── precompute_text_embeddings.py # LLM2Vec embedding extraction
│   ├── generate_g1_motion.py # Kimodo diffusion motion generation
│   ├── validate_g1_motion.py # Structural & physical kinematic validation
│   ├── render_motion_video.py # MuJoCo offscreen MP4 preview renderer
│   ├── gesture_specs.py      # Independent gesture quality specification registry
│   ├── gesture_quality_checker.py # Kinematic fault detector & quality auditor
│   └── gesture_corrector.py  # Targeted joint fault auto-correction engine
│
├── scripts/                  # Tests, setup utilities, historical batch generators & maintenance
│   ├── test_presentation.py # Acceptance test suite (41/41 test cases)
│   ├── test_gesture_quality.py # Quality checker fault injection tests (9 unit tests)
│   ├── test_g1_retargeting_verification.py # Full video retargeting pipeline verification
│   ├── live_full_pipeline_test.py # Live video-to-robot execution test
│   ├── download_models.py    # Downloads required HuggingFace model weights
│   ├── merge_llm2vec_lora.py # Merges LoRA adapters offline for low-RAM inference
│   ├── health_check.py       # System, GPU, RAM, and model diagnostics
│   ├── perfection_pipeline_500_1500.py # Background perfection & publishing state machine
│   └── ...                   # Batch generation & recompute utilities
│
├── data/                     # Authoritative gesture catalogs & manifests
│   ├── gesture_book.json     # 26 core motions
│   ├── gesture_library.json  # Base catalog manifest
│   ├── gesture_library_250.json # 250 core gesture specs & definitions
│   ├── gesture_library_501_1000.json # Gestures 501–1000
│   ├── gesture_library_500_1500.json # Gestures 500–1500
│   └── gesture_library_1501_2000.json # Gestures 1501–2000
│
├── docs/                     # Project documentation & reference guides
│   ├── CLAUDE.md             # Developer & agent architectural guide
│   ├── COMMANDS.md           # Exhaustive CLI command reference
│   ├── PROJECT_BRIEF.md      # High-level project brief & hardware constraints
│   └── all_500_gestures_list.md # Formatted 500 gesture catalog list
│
├── outputs/                  # Generated MP4 previews, qpos CSVs, NPZs, and presentations
├── kimodo_repo/              # Official NVIDIA Kimodo clone (unmodified)
├── venv/                     # Python virtual environment
├── start_all.sh              # Single-command launcher for both backend & frontend
├── start_backend.sh          # Backend service launcher (:8000)
├── start_frontend.sh         # Frontend static server launcher (:8080)
└── run_pipeline.sh           # End-to-end CLI generation pipeline runner
```

## Environment

Isolated conda env `kimodo`; the `base`, `gmr`, and `phmr` envs are untouched.

```bash
conda create -n kimodo python=3.10
conda activate kimodo
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install "numpy>=1.23,<2" "transformers==5.1.0" "peft==0.18.1" "accelerate==1.13.0"
git clone https://github.com/nv-tlabs/kimodo.git kimodo_repo
cd kimodo_repo && pip install -e . && cd ..
pip install mujoco
```

### Why these versions

- **torch 2.4.0+cu121** — Kimodo does not pin torch outside Docker (its official
  Dockerfile inherits torch from `nvcr.io/nvidia/pytorch:24.10-py3`). cu121 was
  chosen because the pre-existing `gmr` and `phmr` envs already run it
  successfully under driver 580.173.02, so it is known-good on this hardware.
- **transformers==5.1.0** — hard pin. Kimodo vendors a patched LLM2Vec
  specifically to work with this version.
- **`pip install -e .` with no extras** — the `[demo]` extra pulls the
  `kimodo-viser` fork for the Gradio/viser web UI, which this CLI-only workflow
  does not use. `[soma]` is for the SOMA skeleton, not G1.
- **mujoco** — required by `kimodo/scripts/mujoco_load.py` but *not* declared in
  `pyproject.toml`, so it must be installed separately.

## Web UI — two commands, two terminals

```bash
# Terminal 1 — BACKEND (models + generation API, port 8000)
cd /home/sai/Desktop/kimodo && ./start_backend.sh

# Terminal 2 — FRONTEND (web UI, port 8080, opens your browser)
cd /home/sai/Desktop/kimodo && ./start_frontend.sh
```

Neither needs `conda activate` — the scripts use the env's Python directly.
Order does not matter: the page shows "backend unreachable" and connects on its
own once the backend is up.

The backend is **supervised**: if it dies it restarts automatically, and a CUDA
fault deliberately exits with code 75 so the fresh process re-probes the GPU
rather than limping on a poisoned context. It also clears a stale listener on
its port first, so a restart never fails with "address already in use".

Ports are overridable: `PORT=9000 ./start_backend.sh` and
`PORT=8081 API=http://127.0.0.1:9000 ./start_frontend.sh`. The frontend writes
the backend address into `.frontend/config.js`, so it lives in one place rather
than being hardcoded in the HTML. CORS on the backend is restricted to loopback
origins.

Single-process mode still works if you prefer one terminal
(`python webapp.py`, UI and API both on 8000).

Type a prompt, get a validated G1 motion with an MP4 and downloadable CSV/NPZ.
Both models stay resident (3.17 GB RSS, 4.42 GB VRAM together on a 6 GB card),
so a request is ~18 s rather than ~30 s of reloading. The encoder is given a
3.5 GiB VRAM budget so the diffusion model still fits alongside it.

GPU work is serialised behind a lock — one card, so a second concurrent request
gets HTTP 429 rather than an OOM. Bound to `127.0.0.1`; `--host` will expose it,
but the app has no auth, so don't.

**Automatic CPU fallback.** At startup the app launches one tiny CUDA kernel
rather than trusting `torch.cuda.is_available()`, which can report True on a
driver that then fails at the first real launch — and False on a wedged
`nvidia_uvm` while `nvidia-smi` still looks healthy. If the probe fails it runs
on CPU instead of refusing to start. Measured cost: 100 steps / 90 frames takes
~116 s on CPU versus 9.8 s on GPU, so drop to ~30 steps for roughly 30 s
results. MuJoCo rendering is unaffected — EGL is graphics, not compute.

**Prompt guard.** A prompt that sanitizes to nothing (`"   "`, `"..."`) reaches
LLM2Vec as an empty sequence and triggers a CUDA `unspecified launch failure`
that poisons the context for the entire process — the server then reports ready
while every generation fails. Such prompts are now rejected with 422 before any
GPU work. The check is Unicode-aware, so non-Latin scripts still pass.

**Gesture coverage is limited by the skeleton, not the prompt.** `g1skel34` has
no finger or thumb joints at all — the most distal arm joints are
`wrist_roll/pitch/yaw`. So "give a thumbs up" renders as a raised hand, and any
prompt whose meaning lives in the fingers cannot be represented. Arm, torso, and
whole-body gestures work well.

## Quick start (CLI)

```bash
conda activate kimodo
cd /home/sai/Desktop/kimodo
./run_pipeline.sh
```

Merges the LoRAs (first run only, ~15 GB), encodes the four default gestures,
generates, clamps, and validates. Idempotent — re-running skips the merge. One
custom prompt: `./run_pipeline.sh "wave with both hands"`.

## Verified results

All four gestures generated locally on the RTX 3050 and validated. 90 frames,
3.000 s @ 30 fps, qpos `(1, 90, 36)`, 100 diffusion steps, seed 42, peak VRAM
1.21 GB, ~9.7 s each.

| gesture | validation | peak vel (rad/s) | peak acc (rad/s²) | right-arm | left-arm | R/L |
|---|---|---|---|---|---|---|
| `right_wave` | PASSED | 9.92 | 199.6 | 5.98 | 0.52 | **11.59** |
| `left_wave` | PASSED ⚠ | 8.31 | 243.1 | 0.57 | 6.90 | **0.08** |
| `right_point` | PASSED | 3.61 | 15.6 | 3.12 | 0.33 | **9.58** |
| `both_raised` | PASSED | 7.42 | 108.1 | 6.44 | 6.99 | **0.92** |

The R/L column is a semantic check, not a structural one: the ratio of total
right-arm to left-arm joint travel. Right-handed prompts are strongly
right-dominant, the left-handed prompt inverts, and the two-handed prompt is
balanced — so the text conditioning is demonstrably doing real work rather than
producing generic motion. For contrast, a random embedding produced 0.09 on the
right-hand prompt.

Encoding is deterministic: the same prompt produced embedding norm 132.50847 in
two independent runs.

⚠ `left_wave` exceeds the acceleration threshold (243 vs 200 rad/s²). It passes
every structural check, but see [Safety](#safety) — that threshold is a
placeholder, not a datasheet figure.

## Measured on this machine

The GPU side was verified by loading `Kimodo-G1-RP-v1` and running real diffusion
sampling with a stub text encoder injected via `load_model(..., text_encoder=...)`,
which isolates the diffusion model from LLM2Vec:

| | |
|---|---|
| Checkpoint load | 2.4 s, resolves to short key `kimodo-g1-rp` |
| Parameters | 283.5 M |
| VRAM, model resident | 1.18 GB |
| VRAM, peak during sampling | **1.21 GB** (100 steps, 90 frames) |
| Sampling time | 9.8 s |
| Output | full field set, 0 NaNs |

1.21 GB against a 6 GB card means **VRAM is not a constraint here** — the earlier
~17 GB figure is almost entirely the text encoder. The full downstream chain
(NPZ → real-robot clamp → qpos CSV → validation) was run on this GPU-generated
motion and passed every structural check.

What remains unverified locally is only the LLM2Vec text embedding itself, for
the RAM reason described above.

## Models

`python download_models.py` fetches four repos (~17.4 GB total):

| Repo | Size | Role |
|---|---|---|
| `nvidia/Kimodo-G1-RP-v1` | 1.1 GB | G1 diffusion checkpoint (g1skel34) |
| `McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp` | 0.17 GB | MNTP LoRA — the `base_model_name_or_path` Kimodo requests |
| `…-mntp-supervised` | 0.17 GB | supervised LoRA layered on top |
| `meta-llama/Meta-Llama-3-8B-Instruct` | 16.1 GB | base weights both LoRAs resolve to |

Two things worth knowing:

- **`meta-llama/Meta-Llama-3-8B-Instruct` is gated.** Accept the license on its
  Hugging Face page and `hf auth login` before downloading. Tokens live in
  `~/.cache/huggingface/`; never commit one.
- **`original/*` is excluded deliberately.** That path holds a 16 GB
  `consolidated.00.pth` duplicate of weights `transformers` never loads.
  Downloading the full snapshot costs 32 GB instead of 17.4 GB.

### If downloads crawl

The `hf_xet` backend degraded badly on this connection — it throttled to
concurrency 1, retried nearly every range request, and reported ~26 KB/s while
writing almost nothing to disk. Disabling it restored ~110 MB/min:

```bash
HF_HUB_DISABLE_XET=1 python download_models.py
```

## Generating a G1 motion (local, two-process)

Because the 8B text encoder and the 283 M diffusion model cannot comfortably
share this machine's memory at the same time, local generation is split into two
processes. The encoder's memory is fully released before Kimodo loads.

```bash
conda activate kimodo
cd /home/sai/Desktop/kimodo

# 1. Encode the prompt(s) once, cache the embeddings.
python precompute_text_embeddings.py --out outputs/embeddings.npz \
    "raise the right hand and wave"

# 2. Generate, clamp to real-robot limits, and write the qpos CSV.
python generate_g1_motion.py "raise the right hand and wave" \
    --embeddings outputs/embeddings.npz --output outputs/right_wave

# 3. Validate.
python validate_g1_motion.py outputs/right_wave/right_wave.csv
```

`generate_g1_motion.py` feeds the cached embedding in through `load_model`'s
`text_encoder=` parameter — a supported Kimodo API — so no Kimodo code is
patched. It applies `apply_g1_real_robot_projection(clamp_to_limits=True)` by
default and writes `.npz`, `.csv`, and a `_meta.json` recording prompt, seed,
steps, timing, and peak VRAM.

**Prompt sanitization matters.** Kimodo runs prompts through
`kimodo.sanitize.sanitize_text` before encoding, so `"raise the right hand and
wave"` becomes `"Raise the right hand and wave."`. Both scripts key the cache on
the sanitized form; a cache built on raw text would miss at generation time.

`precompute_text_embeddings.py` picks a strategy automatically: the stock
CPU path when there is >19 GB of RAM, otherwise an accelerate-sharded load
across GPU + CPU + disk. If the stock sharded loader fails on
`merge_and_unload()` (PEFT will not merge into disk-offloaded modules) it falls
back once to an equivalent unmerged load. Two bounded attempts, never a retry
loop.

## Generating with the stock CLI

```bash
conda activate kimodo
cd /home/sai/Desktop/kimodo
TEXT_ENCODER_DEVICE=cpu TEXT_ENCODER_MODE=local \
  kimodo_gen "raise the right hand and wave" \
  --model Kimodo-G1-RP-v1 \
  --duration 3.0 \
  --num_samples 1 \
  --output outputs/wave/wave
```

- `TEXT_ENCODER_DEVICE=cpu` keeps the 8B LLM2Vec encoder off the GPU. Officially
  documented (CHANGELOG 2026-04-24) and drops VRAM from ~17 GB to <3 GB, which is
  what makes a 6 GB card viable at all.
- `TEXT_ENCODER_MODE=local` skips the probe of the text-encoder HTTP service on
  `127.0.0.1:9550`, which we do not run. Without it, every invocation waits for
  that probe to fail before falling back.
- **The `.csv` is written automatically** for `Kimodo-G1-RP-v1`. `generate.py`
  gates this on the resolved short key being exactly `kimodo-g1-rp`, so
  `Kimodo-G1-SEED-v1` does *not* get an automatic CSV.

### Format notes

- Kimodo NPZ carries `posed_joints`, `global_rot_mats`, `local_rot_mats`,
  `root_positions`, `foot_contacts`; 30 fps, up to 300 frames (10 s).
- G1 CSV is `(T, 36)` MuJoCo qpos: 3 root position + 4 root quaternion (wxyz) +
  29 hinge joints.
- **BVH export does not apply to G1.** `--bvh` is guarded on
  `"somaskel" in skeleton.name` and prints
  `BVH export is only supported for SOMA skeletons` for G1. This is a Kimodo
  design limit, not a setup problem.

## Presentation mode (speech + allowed gestures)

`presentation_compose.py` is what the web app's speech tab runs. It speaks a
paragraph with Edge TTS and performs *allowed* library gestures on the spoken
words while the G1 stays standing in one place:

- `presentation_safety.py` measures every library clip on the real `g1skel34`
  model (foot lift/slide, root drift/yaw, waist ranges, leg travel,
  self-penetration, joint limits) and allows a clip only when everything is
  inside its thresholds and the intent is not a whole-body action. Default
  deny; 113 of 250 clips pass. `PDF_009 Goodbye` is rejected because it steps
  1.27 m backwards, whatever its name says.
- Each gesture's main action starts on its trigger word (measured Edge word
  boundaries, onset-calibrated on the audio; observed lag ≤ 20 ms). When the
  previous gesture is still busy the *speech* pauses at the nearest comma or
  full stop; gestures are never cut short or sped up.
- Between gestures the arms move on C1 paths into the next gesture or into a
  relaxed talking pose (hands at the stomach / lower chest, forearms ~29° below
  horizontal, slow conversational
  motion with still moments) -- never back to attention until the end.
- Every transition is collision-checked, the composed output is re-measured
  and run through `validate_g1_motion.py`, and nothing is written unless it
  passes.

```bash
/home/sai/g1_venv/bin/python presentation_compose.py "Hello! Welcome to robotics. Thank you."
/home/sai/g1_venv/bin/python test_presentation.py      # acceptance suite
```

Restricted words in the text ("walk", "jump", "back flip", ...) are spoken and
listed as *intentionally blocked* in the diagnostics; they never reach the
timeline. The explicit generator tab is the only way to produce such motions,
and nothing here executes on hardware.

## Sentence-to-gesture composition (older path)

`compose_sentence_gestures.py` is a separate, CPU-only retrieval-and-composition
step for conversational playback. It does **not** run Kimodo again. It matches
phrases in a sentence against the existing generated gesture library, adds
idle-to-gesture blends, and writes one final G1 motion archive.

```bash
/home/sai/miniforge3/envs/kimodo/bin/python compose_sentence_gestures.py \
  "Hello everyone, thank you for coming. Please be careful." \
  --output outputs/sentence_sequences/welcome.npz
```

The final file is the requested `.npz`, not a CSV. Its `qpos` field is `(T, 36)`
at 30 fps, and `timeline_json` contains the selected gesture IDs, phrase timing,
and validation result. `speech_offset_s` tells a future TTS/video player when to
start audio, so the first spoken phrase begins with its gesture.

Full library clips last three seconds. When several phrases occur close together,
the planner records pauses in the speech schedule rather than overlapping or
speeding up robot motion. The later TTS stage should honour each event's
`speech_start_s` / `pause_before_s` from `timeline_json`.

Inspect matching without creating output:

```bash
/home/sai/miniforge3/envs/kimodo/bin/python compose_sentence_gestures.py \
  "Welcome. Thank you." --plan-only
```

## Validation

```bash
python validate_g1_motion.py outputs/wave/wave.csv
```

Checks qpos width against `model.nq`, duration/fps, NaN/Inf, root quaternion
normalization, per-joint limits read from the MJCF, finite-difference velocity
and acceleration, forward kinematics on every frame, and root height. Exit code
is 0 only if all structural checks pass.

Visual playback (needs a display):

```bash
python kimodo_repo/kimodo/scripts/mujoco_load.py   # edit the hardcoded motion.csv path first
```

### Clamping to real-robot limits

Kimodo motion can exceed the G1's mechanical limits — NVIDIA's own bundled
example (`assets/demo/examples/kimodo-g1-rp/01_single_text_prompt`) overshoots
`waist_roll_joint` by 0.079 rad and two other joints. The repository ships the
official fix:

```python
from kimodo.exports.mujoco import apply_g1_real_robot_projection
pos, rot = apply_g1_real_robot_projection(skeleton, joints_pos, joints_rot,
                                          clamp_to_limits=True)
```

This projects to 1-DoF-per-hinge and clamps to the MJCF limits. Applying it to
the example above cleared all three violations. Treat it as mandatory before any
motion is considered for hardware.

## Safety

The validator's velocity (10 rad/s) and acceleration (200 rad/s²) thresholds are
**conservative placeholders chosen for this project, not Unitree datasheet
figures.** Replace them with values from the G1 documentation for your specific
hardware revision before relying on them.

Passing validation means the motion is structurally coherent in simulation. It
does not mean it is safe to execute. Before any hardware step you still need
joint-ordering confirmation against the real SDK, unit and coordinate-convention
checks, velocity/acceleration limits from official specs, and a hardware-in-the-
loop review — none of which this repository performs.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `CUDA unknown error` on every process, but `nvidia-smi` looks fine | `nvidia_uvm` is wedged. Check `lsmod \| grep nvidia_uvm` — refcount 0 means no CUDA process holds it and it is safe to reload: `sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm`. Display is unaffected (`nvidia_drm`/`nvidia_modeset` stay loaded). |
| Web app says ready but every generation fails with `unspecified launch failure` | The CUDA context is poisoned and cannot recover in-process. Newer builds detect this, mark the server fatal, and tell you to restart. A prompt that sanitizes to nothing used to cause it; that input is now rejected with 422 before any GPU work. |
| `ModuleNotFoundError: No module named 'viser'` | `kimodo_demo` needs the `[demo]` extra. Not installed by design; `pip install -e ".[demo]"` if you want the web UI. |
| `Error: Kimodo NPZ must contain 'local_rot_mats'` | Input is a demo-viewer NPZ (only `posed_joints`/`global_rot_mats`/`foot_contacts`). Only `kimodo_gen` output has the full field set. |
| Generation waits, then "Text encoder service is unreachable" | Expected without the encoder service. Set `TEXT_ENCODER_MODE=local` to skip the probe. |
| CUDA OOM during generation | Confirm `TEXT_ENCODER_DEVICE=cpu` is actually exported; without it Kimodo needs ~17 GB. |
| Downloads stall with network traffic but no disk growth | `hf_xet` thrashing. Use `HF_HUB_DISABLE_XET=1`. |
| `FutureWarning: torch.load with weights_only=False` | Upstream Kimodo warning from loading skeleton `.p` assets. Harmless. |
| No `.csv` produced | Model resolved to something other than `kimodo-g1-rp`. Only that key auto-exports CSV. |

## Licenses

Kimodo code is Apache-2.0. `Kimodo-G1-RP-v1` is under the NVIDIA Open Model
License. Llama 3 weights are under the Meta Llama 3 Community License, which
governs how the text encoder may be used.
