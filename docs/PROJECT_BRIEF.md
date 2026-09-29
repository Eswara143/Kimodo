# Project Brief — NVIDIA Kimodo → Unitree G1 Conversational Gestures

Copy-pasteable summary of what this project is and where it stands. Written to be
handed to a person or an AI assistant as context.

---

## What we are building

A system that turns spoken/written language into physical gestures for a Unitree
G1 humanoid robot, so the robot can gesture naturally while it talks.

The G1 already speaks. This project adds the **body language**.

## The pipeline

```
text prompt
   ↓  LLM2Vec (Llama-3-8B) text encoder
embedding (4096-d)
   ↓  NVIDIA Kimodo-G1-RP-v1 diffusion model (283 M params)
raw motion (34-joint G1 skeleton, 30 fps)
   ↓  apply_g1_real_robot_projection(clamp_to_limits=True)
motion clamped to the robot's real mechanical limits
   ↓  MujocoQposConverter
MuJoCo qpos CSV  (T, 36)  =  3 root position + 4 root quaternion + 29 joints
   ↓  validate_g1_motion.py   (9 structural checks, exit 0 = pass)
validated motion  →  MP4 preview  →  gesture library
```

Everything is **simulation-side only**. Nothing connects to physical hardware,
and passing validation is explicitly not authorization to run on the robot.

## Hardware it runs on

Ubuntu 22.04 laptop · RTX 3050 **6 GB VRAM** · **16 GB RAM** · driver 580.173.02

The binding constraint turned out to be **system RAM, not VRAM**. The diffusion
model needs only 1.21 GB of VRAM; the 8B text encoder needs ~15 GB, which does
not fit. See "Hard problems solved" below.

## What exists now

**A working local web app** — type a gesture prompt, get a validated G1 motion
with video and downloadable CSV/NPZ, in roughly 10–16 s.

```bash
./start_backend.sh     # models + API on :8000
./start_frontend.sh    # web UI on :8080
# or ./start_all.sh for both
```

**A curated gesture library — 26 motions, all validated**, built from a
conversational catalog of 83 semantic intents. Many intents share one motion
(Hello / Hi / Good morning → one wave), so 26 motions cover **115 trigger
phrases**. Catalogued in `gesture_book.json` with `hand_usage` measured from
actual joint travel rather than asserted.

**A command-line pipeline** for batch work, plus a self-contained Kaggle notebook
as a fallback for machines with more RAM.

## Hard problems solved

**The 8B encoder did not fit.** 15.01 GB against ~13.5 GB of GPU+RAM forces
accelerate to offload part of the model to disk — but PEFT cannot apply a LoRA to
a disk-offloaded module (`KeyError: '<layer>.base_layer.weight'`, from both
`merge_and_unload()` and `PeftModel.from_pretrained`). Fix: fold both LoRAs into
the base weights offline, shard by shard, so PEFT is absent at load time. Peak
memory during the merge is one shard (~5 GB). Encoder now loads in 11 s.

**A prompt that sanitizes to nothing crashed the GPU.** `"   "` reached LLM2Vec
as an empty sequence, raised CUDA `unspecified launch failure`, poisoned the
context process-wide, and left `nvidia_uvm` wedged so no CUDA process on the
machine worked. Such prompts are now rejected before any GPU work, and the
backend exits deliberately on CUDA faults so its supervisor restarts it clean.

**Raw Kimodo output can exceed the robot's joint limits** — NVIDIA's own bundled
example overshoots `waist_roll` by 0.079 rad. Real-robot projection with clamping
is applied to every motion, always.

## Hard limits of the G1 skeleton

`g1skel34` has 29 joints: 12 leg, 3 waist, 14 arm. It has **no neck, head, finger
or thumb joints**. Consequences:

- Thumbs-up, finger counting, OK sign, heart shape: **impossible**
- Nodding and head-shaking: **impossible** (no neck)
- These fail *silently* — they render as a plausible raised hand

13 of the 83 catalog intents hit these limits and were given explicit substitutes
(thumbs-up → applause, nod → torso lean), documented as substitutes rather than
quietly approximated.

## Verification approach

Structural validation alone is not enough — it passes on motion that ignores the
prompt entirely. So each motion is also checked **semantically**: total left-arm
vs right-arm joint travel. A right-handed prompt should be strongly
right-dominant (measured ~11:1), a left-handed prompt inverts (~0.08), two-handed
is balanced (~0.92). A random embedding scores ~0.09 on a right-handed prompt,
which is what makes this a real test of text conditioning.

## Key technical decisions

- **Kimodo is never patched.** Integration goes through its public
  `load_model(..., text_encoder=<obj>)` injection point.
- **Prompts begin with "A person…"** and use medium detail, per NVIDIA's
  documented guidance in `docs/source/key_concepts/limitations.md`.
- **`transformers==5.1.0` is a hard pin** — Kimodo vendors a patched LLM2Vec
  tested only against it.
- Text encoding and diffusion run as **separate processes** in CLI mode so the
  encoder's memory is released first; the web app keeps both resident on purpose,
  which fits only by capping the encoder's VRAM budget.

## What is not done

Physical robot execution. That is deliberately the next phase and needs: the
correct Unitree SDK version for this G1, firmware compatibility, a decision
between low-level and high-level control, and hardware-in-the-loop review.

Also outstanding: the velocity (10 rad/s) and acceleration (200 rad/s²)
thresholds in the validator are **conservative placeholders chosen for this
project, not Unitree datasheet figures**. Seven library gestures already exceed
the acceleration one. Replace them with real values for the specific G1 revision
before any of this informs hardware.
