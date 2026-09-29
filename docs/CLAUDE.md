# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A wrapper around NVIDIA Kimodo that turns a text prompt into a validated Unitree
G1 humanoid motion:

```
text → LLM2Vec encoder → Kimodo-G1-RP-v1 diffusion → NPZ
     → real-robot projection/clamp → MuJoCo qpos CSV → offline validation
```

`kimodo_repo/` is an unmodified clone of `nv-tlabs/kimodo` (Apache-2.0) and is
**not** vendored into version control — see `.gitignore`. Everything at the top
level is this project's own code. The design rule is that Kimodo is never
patched; integration happens through its public API.

Work is **simulation-side only**. Nothing here opens a connection to physical
hardware, and passing validation is explicitly not hardware authorization.

## Commands

All work happens in the isolated `kimodo` conda env. The `base`, `gmr`, and
`phmr` envs belong to unrelated projects — never modify them.

`COMMANDS.md` is the exhaustive command reference for humans; this section is
only what you need to work on the code.

```bash
# Web app: two processes, two terminals. These do NOT need conda activate --
# they invoke the env's interpreter directly.
./start_backend.sh             # models + API on :8000, supervised
./start_frontend.sh            # static UI on :8080, opens a browser

conda activate kimodo          # or prefix: conda run -n kimodo <cmd>

./run_pipeline.sh              # whole thing: merge -> encode -> generate -> validate
python health_check.py         # versions, GPU, models, HF auth, RAM budget
python download_models.py      # fetch the four HF repos (~17.4 GB)
python merge_llm2vec_lora.py   # fold both LoRAs into base weights (once, ~15 GB)
python render_motion_video.py --all   # qpos CSV -> MP4 via MuJoCo offscreen

# Local generation is two processes on purpose (see Architecture)
python precompute_text_embeddings.py --out outputs/embeddings.npz "raise the right hand and wave"
python generate_g1_motion.py "raise the right hand and wave" \
    --embeddings outputs/embeddings.npz --output outputs/right_wave
python validate_g1_motion.py outputs/right_wave/right_wave.csv   # exit 0 == all checks pass

# Sentence + speech: no model load, composes library gestures only (~7 s)
python compose_sentence_gestures.py "Hello everyone, thank you for coming."

# Presentation mode (the web app's speech tab): allowlisted gestures on a fixed stance
python presentation_safety.py --show        # (re)build the allowlist, print the verdict table
python presentation_compose.py "Hello! Welcome to robotics. Thank you."   # -> outputs/presentations/<slug>/
python test_presentation.py                 # acceptance suite, exit 0 == all pass (needs network for Edge TTS)
```

The interpreter the start scripts actually use is `/home/sai/g1_venv/bin/python`
(they fall back to the conda env only if that venv is absent); run the scripts
above with it when `conda activate kimodo` is not available.

Stock Kimodo CLI (single process, needs ~16 GB RAM free for the encoder):

```bash
TEXT_ENCODER_DEVICE=cpu TEXT_ENCODER_MODE=local \
  kimodo_gen "raise the right hand and wave" --model Kimodo-G1-RP-v1 \
  --duration 3.0 --num_samples 1 --output outputs/wave/wave
```

There is no unit-test suite. `validate_g1_motion.py` is the de-facto test: it
exits non-zero on any structural failure, so it works in a CI-style gate. To
exercise the pipeline without the 8B encoder, inject a stub encoder (see
Architecture) — that isolates the diffusion path in a couple of seconds.

Two checks worth running that the validator does not cover:

- **Semantic**, not just structural. Compare total per-joint travel for left vs
  right arm. A right-handed prompt should be strongly right-dominant (measured
  ~11:1), a left-handed prompt inverts (~0.08), two-handed is balanced (~0.92).
  A random embedding gives ~0.09 on a right-handed prompt, so this is what
  distinguishes real text conditioning from plausible-looking noise.
- **Adversarial input against the API.** Do this against a throwaway backend,
  never the one someone is using — an empty-sanitizing prompt used to take the
  GPU down for the whole machine (see gotchas).

## Architecture

### Memory split drives the process boundary

The binding constraint on the target machine (RTX 3050 6 GB, 16 GB RAM) is
**system RAM, not VRAM**:

- Diffusion model: 283 M params, measured **1.21 GB peak VRAM**. Never a problem.
- LLM2Vec text encoder: Llama-3-8B, **~16.06 GB in bf16**. Larger than free RAM.

So `precompute_text_embeddings.py` and `generate_g1_motion.py` are deliberately
separate processes: the encoder's memory is fully released before Kimodo loads.
Do not merge them into one script.

`TEXT_ENCODER_DEVICE=cpu` (official, CHANGELOG 2026-04-24) is what drops GPU use
from ~17 GB to <3 GB, but it moves the problem into RAM.

### Why the LoRAs are merged offline

15.01 GB of encoder against ~13.5 GB of GPU+RAM means accelerate must offload
part of the model to disk — and **PEFT cannot apply a LoRA to a disk-offloaded
module**. It raises `KeyError: '<layer>.base_layer.weight'` both from
`merge_and_unload()` and from `PeftModel.from_pretrained` building its
`offload_index`. Both failure modes were hit and are reproducible.

`merge_llm2vec_lora.py` folds both adapters into the base weights shard by shard
(peak RAM ~5 GB), producing a plain model with no `adapter_config.json` beside
it. transformers then never enters the PEFT path, disk offload is safe, and the
encoder loads in ~11 s at 5.46 GB RSS / 4.54 GB VRAM. `precompute_text_embeddings.py`
auto-detects the merged directory and prefers it.

Two things in that script are load-bearing:

- **`config._name_or_path` must stay `"meta-llama/Meta-Llama-3-8B-Instruct"`.**
  `LLM2Vec.prepare_for_tokenization` applies the Llama-3 chat template only on an
  exact string match; loading from a local directory overwrites it with the path
  and silently changes every embedding.
- **Adapter order is MNTP then supervised**, accumulated in float32 and cast to
  bf16 once, since the supervised LoRA was trained on top of the merged MNTP model.

A failed load also pins several GiB on the GPU that cannot be reclaimed
in-process, so each strategy must run in a fresh process — hence
`sharded-nomerge` is an explicit `--strategy` rather than an in-process retry.

### The web app inverts the process split

`webapp.py` is the exception to the two-process rule above: it holds the encoder
*and* the diffusion model resident at once, which fits only because the encoder
is capped at a 3.5 GiB VRAM budget and offloads the rest (measured together:
3.17 GB RSS, 4.42 GB VRAM on a 6.09 GB card). That buys ~18 s per request
instead of ~30 s of reloading. Keeping both loaded is the whole reason the UI is
usable, so do not "simplify" it back to per-request loading.

Backend (`webapp.py`, FastAPI) and frontend (`webui.html`, served statically) run
on separate ports. The frontend learns the API address from a generated
`.frontend/config.js`, so the backend URL lives in one place rather than being
baked into the HTML; CORS on the backend is restricted to loopback origins.

Load-bearing details in `start_backend.sh` and `webapp.py`:

- **The supervisor loop restarts the backend on any non-zero exit.** A CUDA fault
  deliberately calls `os._exit(75)` because a poisoned context cannot be repaired
  in-process — the fresh process re-probes the GPU and falls back to CPU if the
  driver is still unhappy. Self-healing depends on both halves; do not remove
  either.
- **`flock` gives single-instance semantics.** The script also kills a stale
  listener on its port at startup, so two supervisors would each kill the other's
  child forever. The lock is what prevents that.
- **`_pick_device()` launches a real kernel** rather than trusting
  `torch.cuda.is_available()`, which lies in both directions (see gotchas).
- **The GPU lock queues (`acquire(timeout=900)`) rather than rejecting.** One
  card, so work is serialised, but a second caller waits instead of getting 429.
- **`/api/gallery`** rebuilds the result list from `outputs/web/*/meta.json` on
  disk, skipping directories with no `.mp4` (crashed or still rendering). Results
  therefore survive reloads and restarts; the UI has no client-side persistence.
  It also walks `outputs/gesture_library/` and enriches `PDF_*` entries from
  `gesture_library_250.json`, which is what the collection filters in the UI
  group on.

### The sentence + speech path never touches the model

`/api/compose-sentence` is the second half of the web app and deliberately
shares nothing with the generator but the output directory conventions. It
matches a sentence against gestures that were *already* generated, so it holds
its own `_sentence_lock` rather than the GPU lock and stays available while the
models are still loading -- which is most of the first minute after a restart.

- **The gesture catalog is the intersection of a manifest and what is on disk.**
  `load_catalog()` in `compose_sentence_gestures.py` prefers
  `gesture_library_250.json` but keeps only entries whose CSV actually exists
  under `outputs/web/pdf_250_library/<id>/`, falling back to the 26-gesture
  `gesture_book.json` when none do. A half-finished batch therefore degrades to
  fewer gestures instead of erroring on a missing file.
- **Gesture timing comes from Edge TTS, not from words-per-minute.**
  `word_timings()` asks `edge_tts` for `WordBoundary` events while synthesising,
  so each gesture is placed where its own trigger phrase is really spoken. The
  `words_per_minute` field sets the Edge speaking `rate` (150 = native pace)
  and is the proportional fallback only when synthesis fails.
- **The motion is authoritative; the speech yields.** Library clips are 3 s and
  trigger words are often under 1 s apart, so the two cannot both be honoured
  as-is. `compose_natural` never time-scales or drops a gesture: every match
  plays in full at 1.0x starting on (or at most `lag_tolerance_s` after) its
  word, and when the next trigger would arrive too early a pause is inserted
  into the *audio* (`insert_speech_pauses`, silence between words -- no pitch
  or rate change). The pause goes on the nearest comma/full stop between the
  two triggers when one exists, so it sounds like a speaker breathing rather
  than stalling mid-phrase. A 26-word sentence with 9 matches therefore runs
  ~31 s against ~10 s of raw speech; that is by design, not a bug. The old
  `compose_synced` (scale clips into the word window, drop what does not fit)
  is what read as "fast and skipping" and is no longer wired to the API.
- **No gesture cap by default.** `find_matches(max_gestures=None)` gestures
  every matched phrase. A cap silently drops the tail of a sentence, which
  users read as skipping; the only remaining limit is `MAX_SEQUENCE_SECONDS`
  (180 s), reported per gesture in `skipped_gestures`.
- **Seams hold the last pose.** When the gap between clips is longer than the
  blend, the frames before the blend must be filled with the previous clip's
  final pose; the array is initialised to idle, and leaving the gap at idle
  produced a 61 rad/s snap at every seam.
- **Do not mux with `apad` + `-shortest` and a copied video stream.** ffmpeg
  keeps padding audio after the last video packet and never exits (it ran 10+
  minutes here and the request hung). State the length explicitly instead:
  `-af apad=whole_dur=<motion_s> -t <motion_s>`, with a `timeout` on the call.
- **Composition is validated before anything is written.** `validate()` runs the
  same `validate_g1_motion.py` checks over the blended result, and a failure
  raises rather than writing a half-valid NPZ -- blending two individually legal
  clips can still cross a joint limit at the seam.
- **The MP4 is a preview; the NPZ is the artifact.** Speech is muxed into the
  video with `ffmpeg -c:v copy` purely so the browser can play both together.
  The downloadable NPZ carries the qpos and the timeline, not the audio.
- **Edge TTS needs network access.** It is a Microsoft cloud service, so the
  sentence tab is the one part of this project that is not fully offline. Both
  TTS calls are wrapped so a failure degrades to a silent video rather than a
  500.

### Presentation mode: default-deny allowlist on one stance

The web app's speech tab is `presentation_*.py`, which supersedes the old
`compose_sentence_gestures.py` path (still served at `/api/compose-sentence`
for the CLI). Presentation text is speech content, never a command; the design
rests on four decisions:

- **Clips are admitted by measurement, not by name.** `presentation_safety.py`
  runs every library clip through MuJoCo FK on the real `g1skel34` model and
  allows it only when foot lift, foot slide, root drift, root yaw, waist
  ranges, leg travel and self-penetration all stay under `THRESHOLDS`, and its
  intent is not in `RESTRICTED_INTENTS`. 113 of 250 pass. `PDF_009 Goodbye`
  steps 1.27 m backwards and is the canonical misleadingly named clip. The
  verdicts are cached in `outputs/presentations/allowlist.json` keyed on CSV
  mtime/size.
- **Self-collision limits are per pair class.** The MJCF hand capsules are 5 cm
  radius against a ~1.5 cm real palm, so two touching palms register ~7 cm of
  overlap. `PAIR_PENETRATION` allows 7.5 cm hand~hand, 5.5 cm hand~other,
  3 cm otherwise; a single 3 cm limit would deny every two-handed gesture.
- **Velocity/acceleration are warnings, not denials.** The validator's
  10 rad/s / 200 rad/s² are documented placeholders that it itself reports as
  warnings; the allowlist denies only above 2x those and surfaces the rest as
  `review_warning`. Transitions the composer generates are held to the strict
  figures. `PDF_001 Hello` (12 rad/s) depends on this.
- **Composed clips play on the attention stance.** Different Kimodo clips stand
  slightly differently; a transition between two of them slid a foot 4 cm. The
  composer therefore keeps root + legs at the verified attention pose and takes
  only waist + arms from each (already-admitted) clip, so feet never move.

Scheduling: each gesture's *main action* (from `analyze_phases`, the span where
the arms are at least halfway to their furthest pose) starts on its trigger
word (tolerance +250/-100 ms); when the previous gesture is still busy the speech
pauses at the last comma/full stop between the two triggers, and a cue with no
such boundary is reported as `timing_conflict` and spoken only. Main actions are
never retimed. Between gestures the composer uses C1 Hermite paths straight
into the next main action or into the relaxed talking track (`IdleTrack`, a
deterministic function of the frame index, so exits/entries can target its
exact pose and velocity) -- the clips' own drop-to-the-sides recoveries are
not used, which is what keeps the hands up between phrases. Every transition is
collision-checked; if it fails it is re-routed through the talking pose, and if
that fails the gesture is dropped with an `invalid_transition` diagnostic and
the plan is rebuilt.

The talking pose is `TALKING_ARMS` (elbow 0.10: forearms ~29° below horizontal,
hands ~4 cm above the v1 pose that read as hanging at the crotch) and the idle
amplitude is `IDLE_MOTION` x `IDLE_AMPLITUDE_SCALE` (1.25); both were set by
measuring FK on the model, not by eye -- the elbow joint reads 0 at ~90° flexion
and *grows* as the forearm drops, same sign on both arms.

Speech: Edge TTS `WordBoundary` offsets lead the audible onset by ~130 ms
(measured); `calibrate_timings` snaps words that follow a silence to the
detected energy onset and shifts the rest by the median. The muxed MP4's audio
matches the schedule within ~1 ms. Pauses are inserted into the WAV as silence
(no time-stretching); the schedule's times are all in the final timeline.

The worker runs as a subprocess (`presentation_worker.py`) with
`RLIMIT_DATA` 4/5 GiB, 2 BLAS threads, `nice 5` and a paced EGL render.
**`RLIMIT_AS` cannot be used** -- the NVIDIA EGL driver maps more virtual
space than that and aborts with `libc++abi: terminating`. VRAM stays above 60 %
of the card because the generator's models (4.3 GB) are resident; the
presentation itself adds ~120 MB.

`FALLBACK_01`/`FALLBACK_02` under `outputs/web/presentation_fallbacks/` are the
talking-idle and attention-hold fallbacks exported as library entries
(`presentation_compose.py --export-fallbacks`); they sit outside the `PDF_`
number space, which belongs to the gesture PDFs (1-250 and 251-500).
Presentations are numbered `PRES_001` onwards from
`outputs/presentations/counter.json`.

### Integration points with Kimodo (do not patch Kimodo)

- **`load_model(..., text_encoder=<obj>)`** — the supported injection point.
  Passing an object skips LLM2Vec instantiation entirely. Any object with
  `__call__(text) -> (tensor, lengths)`, `.to()`, `.eval()`, and `.llm_dim`
  works. `PrecomputedTextEncoder` in `generate_g1_motion.py` uses this for
  cached embeddings; a zero-returning stub uses it for fast pipeline smoke tests.
  The contract: a list yields `(N, 1, 4096)`, a bare string yields `(1, 4096)`.
- **`apply_g1_real_robot_projection(..., clamp_to_limits=True)`** from
  `kimodo.exports.mujoco` — projects to 1-DoF-per-hinge and clamps to MJCF
  limits. Treat as mandatory: raw Kimodo output can exceed the G1's mechanical
  limits, and NVIDIA's own bundled example violates three joints.
- **`MujocoQposConverter.dict_to_qpos` / `.save_csv`** — NPZ dict to `(T, 36)`
  qpos.

### Non-obvious behaviours that will bite

- **A prompt that sanitizes to nothing takes down the GPU.** `"   "` or `"..."`
  reaches LLM2Vec as an empty sequence and raises CUDA `unspecified launch
  failure`, which poisons the context for the entire process — the server then
  reports ready while every request fails, and killing it can leave `nvidia_uvm`
  wedged so *no* CUDA process on the machine works until
  `sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm`. `webapp.py` rejects such
  prompts with 422 before touching the GPU. Keep that guard, and keep it
  Unicode-aware (`\W`, not `[^a-zA-Z0-9]`) so non-Latin scripts still pass.
- **`torch.cuda.is_available()` is not trustworthy.** It returns True on a driver
  that fails at the first real kernel launch, and False on a wedged `nvidia_uvm`
  while `nvidia-smi` still reports a healthy idle GPU. Probe with an actual
  allocation plus `synchronize()`.
- **MuJoCo EGL rendering does not need CUDA** — it is graphics, not compute, so
  video rendering keeps working when the compute stack is broken. `osmesa` is not
  functional in this env; `egl` and `glfw` both are.
- **`g1skel34` has no finger or thumb joints.** The most distal arm joints are
  `wrist_roll/pitch/yaw`. "Thumbs up" can only render as a raised hand, and any
  gesture whose meaning lives in the fingers is unrepresentable. Not fixable by
  prompting.
- **Prompts are sanitized before encoding.** `kimodo.sanitize.sanitize_text`
  turns `"raise the right hand and wave"` into `"Raise the right hand and
  wave."`. Any embedding cache must be keyed on the sanitized form or lookups
  miss at generation time.
- **`device` must be a primitive string.** `load_model` routes it through
  OmegaConf, so `torch.device('cuda')` raises `UnsupportedValueType`. Pass
  `'cuda'`.
- **CSV auto-export is gated on the resolved short key being exactly
  `kimodo-g1-rp`.** `Kimodo-G1-SEED-v1` resolves to `kimodo-g1-seed` and gets no
  automatic CSV.
- **Bundled demo NPZs are a reduced format** (`posed_joints`,
  `global_rot_mats`, `foot_contacts` only). `kimodo_convert --to g1-csv` needs
  `local_rot_mats` and `root_positions`, which only `kimodo_gen` output has.
  Derive the missing fields with `global_rots_to_local_rots` if converting one.
- **BVH export does not apply to G1** — guarded on `"somaskel" in
  skeleton.name`. Not a bug to fix.
- **Postprocessing is disabled for G1** in `generate.py`; match that when
  calling the model directly.
- **`mujoco` is not declared in Kimodo's `pyproject.toml`** despite
  `scripts/mujoco_load.py` importing it. Install separately.
- **`hf_xet` can thrash** (throttles to ~26 KB/s with network traffic but no
  disk growth). `HF_HUB_DISABLE_XET=1` restores normal throughput.

### Model dependency chain

Kimodo asks for `McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp` as its
*base*, which is only a 168 MB LoRA. Its `adapter_config.json` resolves the real
weights to the gated `meta-llama/Meta-Llama-3-8B-Instruct`, and a second
supervised LoRA layers on top. All four repos are required. Exclude
`original/*` from the Llama download — it is a 16 GB `.pth` duplicate that
`transformers` never loads, and including it doubles the download to 32 GB.

## Version pins

`transformers==5.1.0` is a hard pin: Kimodo vendors a patched LLM2Vec tested
only against it. `torch 2.4.0+cu121` was chosen because it is already proven on
this machine's driver; Kimodo itself does not pin torch outside Docker (its
Dockerfile inherits it from `nvcr.io/nvidia/pytorch:24.10-py3`). Install Kimodo
with no extras — `[demo]` pulls the viser/Gradio UI this CLI workflow does not
use, `[soma]` is for a different skeleton.

## Fallback environment

`kaggle_kimodo_g1.py` holds a cell-by-cell Kaggle notebook for when local RAM is
insufficient (Kaggle has ~30 GB). It authenticates from a Kaggle secret, avoids
inbound ports, and runs the same generate → clamp → validate chain.
