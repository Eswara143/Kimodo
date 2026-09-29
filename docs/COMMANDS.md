# COMMANDS — Kimodo-G1

Every command for this project, in one place. Project root is
`/home/sai/Desktop/kimodo` — all paths below are relative to it.

The `./start_*.sh` scripts do **not** need `conda activate`; they call the env's
Python directly. Everything else does, unless you use the full interpreter path
`/home/sai/miniforge3/envs/kimodo/bin/python`.

Simulation only. Nothing here connects to a physical robot.

---

## 1. Daily use — the web UI

Two terminals:

```bash
# Terminal 1 — BACKEND (models + generation API, port 8000)
cd /home/sai/Desktop/kimodo && ./start_backend.sh

# Terminal 2 — FRONTEND (web UI, port 8080, opens your browser)
cd /home/sai/Desktop/kimodo && ./start_frontend.sh
```

Then open **http://127.0.0.1:8080**

The page has two tabs:

- **Generate a Gesture** — the diffusion pipeline. One text prompt in, one new
  validated G1 motion out (~20-35 s, needs the models loaded).
- **Sentence + Synced Gestures** — speech. Type a whole sentence, pick an Edge
  TTS voice, and it is spoken aloud while gestures from the pre-generated
  library are played at the moment their own trigger words are said. No
  diffusion and no GPU, so it answers in a few seconds and works even while
  the models are still loading.

Order does not matter — the page says "backend unreachable" and connects on its
own once the backend is up. Ctrl-C in each terminal stops them.

The backend restarts itself if it crashes, and only one instance can run at a
time (an `flock` guard — a second launch is refused instead of two supervisors
fighting over the port).

### Different ports

```bash
PORT=9000 ./start_backend.sh
PORT=8081 API=http://127.0.0.1:9000 ./start_frontend.sh
```

### Single terminal instead of two

```bash
conda activate kimodo
python webapp.py                  # UI and API both on http://127.0.0.1:8000
python webapp.py --port 8080
```

### Stop everything

```bash
pkill -f start_backend.sh; pkill -f webapp.py; pkill -f 'http.server 8080'
```

---

## 2. One-time setup

Only needed on a fresh machine, or if the caches are deleted.

```bash
# Environment (isolated; base/gmr/phmr are never touched)
conda create -n kimodo python=3.10
conda activate kimodo
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install "numpy>=1.23,<2" "transformers==5.1.0" "peft==0.18.1" "accelerate==1.13.0"
git clone https://github.com/nv-tlabs/kimodo.git kimodo_repo
cd kimodo_repo && pip install -e . && cd ..
pip install mujoco

# Hugging Face login (browser device flow — never paste a token into a terminal
# that is being shared, and never into a file)
hf auth login
hf auth whoami                    # prints username only

# Models — ~17.4 GB. HF_HUB_DISABLE_XET=1 avoids the xet backend, which
# throttled to ~26 KB/s here; classic HTTP sustained ~106 MB/min.
HF_HUB_DISABLE_XET=1 python download_models.py

# Fold both LoRAs into the base weights — REQUIRED on this 16 GB machine.
# Writes ~15 GB to ~/.cache/kimodo/llm2vec_merged. Skips if already built.
python merge_llm2vec_lora.py
python merge_llm2vec_lora.py --force      # rebuild from scratch
```

`meta-llama/Meta-Llama-3-8B-Instruct` is gated: accept its licence on the model
page with the same HF account first, or the download 401s.

---

## 3. Command-line pipeline (no web UI)

### Everything at once

```bash
conda activate kimodo
./run_pipeline.sh                              # merge → encode → generate → validate
./run_pipeline.sh "wave with both hands"       # one custom prompt
GPU_GIB=4 CPU_GIB=6 ./run_pipeline.sh          # tighter memory budgets
```

### Step by step

```bash
# 1. Encode prompt(s) — several at once amortises the model load
python precompute_text_embeddings.py \
    "raise the right hand and wave" \
    "raise the left hand and wave" \
    --out outputs/gesture_embeddings.npz --gpu-gib 5 --cpu-gib 7

# 2. Generate + clamp + write qpos CSV
python generate_g1_motion.py "raise the right hand and wave" \
    --embeddings outputs/gesture_embeddings.npz \
    --output outputs/right_wave --duration 3.0 --diffusion-steps 100 --seed 42

# 3. Validate — exit code 0 means every structural check passed
python validate_g1_motion.py outputs/right_wave/right_wave.csv
echo $?

# 4. Render to MP4
python render_motion_video.py outputs/right_wave/right_wave.csv
python render_motion_video.py --all            # every gesture in outputs/
```

Useful flags: `--no-clamp` on step 2 skips the real-robot projection (produces
motion that may exceed the robot's limits — not recommended), and
`--strategy {auto,cpu,sharded,sharded-nomerge,merged}` on step 1 forces an
encoder loading path instead of auto-detecting the merged one.

### Stock Kimodo CLI

Needs ~16 GB of free RAM for the unmerged encoder, so it does not fit on this
machine — kept here for reference and for Kaggle.

```bash
TEXT_ENCODER_DEVICE=cpu TEXT_ENCODER_MODE=local \
  kimodo_gen "raise the right hand and wave" --model Kimodo-G1-RP-v1 \
  --duration 3.0 --num_samples 1 --seed 42 --output outputs/wave/wave

kimodo_convert input.npz output.csv --to g1-csv
```

---

## 3b. Presentation mode — speech + allowed gestures (web tab)

The **Presentation: Speech + Gestures** tab. Type a paragraph, pick a voice,
speech rate and phrase pause, click **Prepare Presentation**, then Play /
Pause / Resume / Stop. The robot stays standing in place: only clips that pass
the measured stationary allowlist (113 of 250) can enter the timeline, and
walking / jumping / kicking / flips / turns / floor motions are never performed
from presentation text ("Our robot can walk" is spoken, not executed). The
page shows the current phrase, the gesture timeline, validation results and an
Errors & Diagnostics list that distinguishes blocked actions, no-match phrases,
missing assets, timing conflicts, invalid transitions and speech failures.

```bash
PY=/home/sai/g1_venv/bin/python
$PY presentation_safety.py --show            # allowlist table (cached in outputs/presentations/allowlist.json)
$PY presentation_compose.py "Hello! Welcome to robotics. Thank you." --voice en-US-AndrewNeural --wpm 130
$PY presentation_compose.py --export-fallbacks   # FALLBACK_01 talking idle, FALLBACK_02 attention hold
$PY test_presentation.py                     # acceptance suite (Edge TTS needs network)

curl -X POST http://127.0.0.1:8000/api/presentation/prepare -H 'Content-Type: application/json' \
    -d '{"text": "Hello! Thank you.", "voice": "en-US-AvaNeural", "words_per_minute": 150, "phrase_pause_s": 0.8}'
curl http://127.0.0.1:8000/api/presentation/jobs/<job_id>     # progress, then result + worker resource peaks
curl http://127.0.0.1:8000/api/presentation/gallery
curl http://127.0.0.1:8000/api/presentation/allowlist
```

Outputs land in `outputs/presentations/PRES_<n>/`: `preview.mp4` (motion +
speech), `PRES_<n>.npz` (qpos + schedule), `.csv`, `speech.wav`, `schedule.json`.

## 3c. Sentence + speech from the pre-generated library (older path)

Matches a sentence against the gesture library on disk, speaks it with Microsoft
Edge neural TTS, and lays each gesture on the timeline at the moment its trigger
phrase is actually spoken (word boundaries come back from Edge TTS itself, so the
timing is measured rather than estimated from a words-per-minute figure). The
result is one combined, structurally validated `(T, 36)` NPZ plus an MP4 preview
with the speech muxed in.

Every matched word gets its gesture, played in full at natural speed -- nothing
is sped up and nothing is skipped. Library clips are 3 s each, so when the next
gesture word would arrive before the current gesture is done the *voice* pauses
(at the nearest comma or full stop when there is one) until the robot is ready.
Expect a sentence with many gesture words to take roughly 3.5 s per gesture:
9 gestures is about 31 s of video from 10 s of raw speech. The only hard limit
is 180 s of total motion; anything past it is listed under "skipped" with the
reason. "Speech pace" changes the Edge speaking rate (150 = the voice's own
pace); slower speech means fewer and shorter inserted pauses.

Nothing here touches the diffusion model or the GPU — it only replays motions
that were generated earlier.

### Web UI

No longer wired to the UI (the Presentation tab replaced it); the API and CLI
below still work. Voices include US, UK, Indian English and Telugu
(`te-IN-ShrutiNeural`, `te-IN-MohanNeural`).

### Command line

```bash
conda activate kimodo

python compose_sentence_gestures.py "Hello everyone, thank you for coming."
```

### API

```bash
curl -X POST http://127.0.0.1:8000/api/compose-sentence \
    -H 'Content-Type: application/json' \
    -d '{"sentence": "Hello everyone, thank you for coming. Please be careful.",
         "max_gestures": 4, "voice": "en-US-AvaNeural"}'

curl "http://127.0.0.1:8000/api/voices"                     # list voices
curl "http://127.0.0.1:8000/api/tts?text=Hello" -o out.mp3  # speech only
```

A sentence whose words match nothing in the library still composes — it just
returns an idle pose for the spoken duration.

Only one backend can run on this machine at a time. The `kimodo (copy)` folder
has its own on port 8001; starting both makes them fight for the 6 GB card and
16 GB of RAM, and both get OOM-killed (exit 137).

---

## 4. Checks and diagnostics

```bash
conda activate kimodo
python health_check.py            # versions, GPU, models, HF auth, memory

nvidia-smi
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
free -h
df -h /

# Is the backend alive and what device is it on?
curl -s http://127.0.0.1:8000/api/status

# Everything generated so far
curl -s http://127.0.0.1:8000/api/gallery | python3 -m json.tool | head -40

# Generate straight from the API
curl -X POST http://127.0.0.1:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"raise the right hand and wave","duration":3.0,"steps":100,"seed":42}'

# Logs
tail -f backend.log
tail -f frontend.log
```

---

## 5. Troubleshooting

### GPU is wedged — `CUDA unknown error` everywhere but `nvidia-smi` looks fine

The app auto-falls back to CPU so you are never blocked, but CPU is ~12× slower.
Check the refcount is 0 first (it means no CUDA process holds the module, so
reloading is safe and the display is unaffected):

```bash
lsmod | grep nvidia_uvm
sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm
```

Then restart the backend terminal — it re-probes the GPU and switches back on its
own. If `rmmod` says the module is in use, a reboot does the same job.

### "address already in use"

```bash
pkill -f webapp.py; sleep 3; ./start_backend.sh
```

### "another backend supervisor already holds the lock"

One is already running. Stop it, or use a different `PORT=`.

```bash
pkill -f start_backend.sh
```

### Generation feels slow

Check the status pill in the UI. **CPU mode** means ~80 s per gesture — drop
diffusion steps to **30** for roughly 30 s. GPU mode is ~10 s at 100 steps.

### Other known behaviours

| Symptom | Cause |
|---|---|
| Prompt rejected with 422 | It sanitizes to fewer than two word characters. Empty prompts used to crash CUDA, so they are blocked before any GPU work. |
| "thumbs up" looks like a raised hand | `g1skel34` has no finger or thumb joints. Hardware-model limit, not a bug. |
| `--bvh` prints "SOMA skeletons only" | BVH export does not apply to G1. Expected. |
| No `.csv` from `kimodo_gen` | Only `Kimodo-G1-RP-v1` (short key `kimodo-g1-rp`) auto-exports CSV. G1-SEED does not. |
| `FutureWarning: torch.load ... weights_only` | Upstream Kimodo warning loading skeleton assets. Harmless. |

---

## 6. Kaggle (free ~30 GB RAM alternative)

`kaggle_kimodo_g1.py` holds a self-contained 7-cell notebook — nothing to upload
alongside it. In the Kaggle UI: Accelerator → GPU, Internet → On, and add
`HF_TOKEN` under Add-ons → Secrets.

```bash
python kaggle_kimodo_g1.py        # prints the cells to paste
```

With ~30 GB of RAM the stock CPU encoder path loads normally, so the LoRA merge
is unnecessary there.

---

## 7. Where things live

| Path | What |
|---|---|
| `outputs/<gesture>/` | CLI runs — `.npz`, `.csv`, `_meta.json` |
| `outputs/web/<name>/` | Web UI runs — same plus `.mp4` |
| `outputs/web/pdf_250_library/` | 90 pre-generated library gestures the sentence tab draws on |
| `outputs/sentence_sequences/<name>/` | Sentence runs — combined `.npz`, `speech.mp3`, `preview.mp4` |
| `outputs/videos/` | `render_motion_video.py --all` output |
| `~/.cache/kimodo/llm2vec_merged/` | Merged encoder, ~15 GB |
| `~/.cache/huggingface/` | Downloaded models, ~17.4 GB |
| `kimodo_repo/` | Official NVIDIA clone, commit `1aece8c` — never patched |
| `backend.log`, `frontend.log` | Service logs |
