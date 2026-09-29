"""Kimodo-G1 gesture generation on a Kaggle GPU notebook.

Paste each CELL_n block into its own Kaggle cell, in order. The notebook is
self-contained: nothing needs to be uploaded alongside it.

WHY KAGGLE. Kimodo's text encoder is LLM2Vec over Meta-Llama-3-8B. The
bidirectional encoder body is 15.01 GB in bfloat16. On a 16 GB laptop that does
not fit in RAM, and the obvious workaround -- letting accelerate offload part of
the model to disk -- is incompatible with PEFT: applying a LoRA adapter over
disk-offloaded modules raises KeyError on '<layer>.base_layer.weight', both when
merging (merge_and_unload) and when stacking the second adapter
(PeftModel.from_pretrained). Kaggle has ~30 GB of RAM, so the stock CPU path
loads normally and none of that applies.

The diffusion model itself is small: 283 M params, measured 1.21 GB peak VRAM on
an RTX 3050. VRAM was never the constraint; system RAM was.

SETUP (once, in the Kaggle UI):
  Notebook settings -> Accelerator -> GPU
  Notebook settings -> Internet -> On
  Add-ons -> Secrets -> add HF_TOKEN  (never paste a token into a cell)
  Accept the license on huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
  with the same account, or the download 401s.

Outputs land in /kaggle/working/outputs and are downloadable from the Output tab
once the notebook is committed.
"""

# =============================================================================
# CELL 1 - Inspect the assigned hardware before installing anything.
# Kaggle's GPU type, RAM, and disk are not guaranteed. The hard requirement is
# RAM: the encoder needs ~15 GB resident. Any CUDA GPU is sufficient.
# =============================================================================
CELL_1 = r"""
!nvidia-smi
!free -h
!df -h /kaggle/working
import sys, torch, psutil
print("python :", sys.version.split()[0])
print("torch  :", torch.__version__, "| cuda:", torch.version.cuda)
print("cuda ok:", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("gpu    :", p.name, f"{p.total_memory/1e9:.1f} GB")
ram = psutil.virtual_memory().total / 1e9
print("ram    :", f"{ram:.1f} GB")
assert ram > 20, f"Need >20 GB RAM for the 15 GB encoder; this session has {ram:.1f} GB"
print("OK: enough RAM for the CPU text encoder")
"""

# =============================================================================
# CELL 2 - Install Kimodo.
# Kaggle images ship a working torch/CUDA pairing; do NOT reinstall torch.
# transformers==5.1.0 is a hard pin -- Kimodo vendors a patched LLM2Vec tested
# only against it. No [demo] extra: the viser/Gradio UI is unreachable from a
# Kaggle notebook anyway, and mujoco is needed but is not declared in Kimodo's
# pyproject.toml despite scripts/mujoco_load.py importing it.
# =============================================================================
CELL_2 = r"""
import torch
before = torch.__version__

!pip install -q "transformers==5.1.0" "peft==0.18.1" "accelerate==1.13.0"
!git clone -q https://github.com/nv-tlabs/kimodo.git /kaggle/working/kimodo_repo
!pip install -q -e /kaggle/working/kimodo_repo
!pip install -q mujoco

import importlib, torch
importlib.reload(torch)
print("torch before:", before, "-> after:", torch.__version__)
print("cuda ok:", torch.cuda.is_available())
assert torch.cuda.is_available(), "torch lost CUDA - do not reinstall torch in this notebook"
"""

# =============================================================================
# CELL 3 - Authenticate from the Kaggle secret. Prints the username only.
# =============================================================================
CELL_3 = r"""
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login, whoami

login(token=UserSecretsClient().get_secret("HF_TOKEN"), add_to_git_credential=False)
print("logged in as:", whoami()["name"])
"""

# =============================================================================
# CELL 4 - Fetch the four required repos (~17.4 GB).
#
# Kimodo asks for the MNTP repo as its "base model", but that repo is only a
# 168 MB LoRA; its adapter_config.json resolves the real weights to the gated
# meta-llama repo, and a second supervised LoRA layers on top. All four needed.
#
# original/* is excluded: a 16 GB consolidated .pth duplicate that transformers
# never loads. Including it doubles the download to 32 GB.
#
# HF_HUB_DISABLE_XET=1 because the xet backend throttled to ~26 KB/s with heavy
# retries during local testing; the classic HTTP path sustained ~106 MB/min.
# =============================================================================
CELL_4 = r"""
import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import snapshot_download

for repo, ignore in [
    ("nvidia/Kimodo-G1-RP-v1", None),
    ("McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp", None),
    ("McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised", None),
    ("meta-llama/Meta-Llama-3-8B-Instruct", ["original/*"]),
]:
    print("downloading", repo, flush=True)
    snapshot_download(repo_id=repo, ignore_patterns=ignore)

# Verify the big shards arrived intact rather than trusting the loop finished.
import os
from huggingface_hub import snapshot_download
p = snapshot_download("meta-llama/Meta-Llama-3-8B-Instruct",
                      local_files_only=True, ignore_patterns=["original/*"])
expected = {"model-00001-of-00004.safetensors": 4976698672,
            "model-00002-of-00004.safetensors": 4999802720,
            "model-00003-of-00004.safetensors": 4915916176,
            "model-00004-of-00004.safetensors": 1168138808}
for f, want in expected.items():
    got = os.path.getsize(os.path.realpath(os.path.join(p, f)))
    assert got == want, f"{f}: {got:,} != {want:,}"
print("all shards verified")
"""

# =============================================================================
# CELL 5 - Generate gestures with the REAL text encoder.
#
# TEXT_ENCODER_DEVICE=cpu keeps the 15 GB encoder off the GPU (official,
# CHANGELOG 2026-04-24), dropping VRAM from ~17 GB to <3 GB.
# TEXT_ENCODER_MODE=local skips a probe of the encoder HTTP service on port
# 9550, which this notebook does not run; without it every call waits for that
# probe to time out before falling back.
#
# Kimodo-G1-RP-v1 writes the MuJoCo qpos CSV automatically. generate.py gates
# that on the resolved short key being exactly "kimodo-g1-rp", so
# Kimodo-G1-SEED-v1 would produce no CSV.
# =============================================================================
CELL_5 = r"""
import os, subprocess, time
os.environ["TEXT_ENCODER_DEVICE"] = "cpu"
os.environ["TEXT_ENCODER_MODE"] = "local"

GESTURES = {
    "right_wave":  "raise the right hand and wave",
    "left_wave":   "raise the left hand and wave",
    "right_point": "point forward with the right hand",
    "both_raised": "raise both hands above the head",
}

os.makedirs("/kaggle/working/outputs", exist_ok=True)
for name, prompt in GESTURES.items():
    out = f"/kaggle/working/outputs/{name}/{name}"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    print(f"=== {name}: {prompt!r} ===", flush=True)
    t0 = time.time()
    subprocess.run([
        "kimodo_gen", prompt,
        "--model", "Kimodo-G1-RP-v1",
        "--duration", "3.0",
        "--num_samples", "1",
        "--seed", "42",
        "--output", out,
    ], check=True)
    print(f"  {time.time()-t0:.1f}s", flush=True)

!find /kaggle/working/outputs -type f | sort
"""

# =============================================================================
# CELL 6 - Clamp to real-robot limits, then validate.
#
# Raw Kimodo output can exceed the G1's mechanical limits: NVIDIA's own bundled
# example overshoots waist_roll_joint by 0.079 rad plus two other joints.
# apply_g1_real_robot_projection is the repository's official fix and should be
# treated as mandatory before a motion is considered for hardware.
#
# The velocity/acceleration ceilings below are CONSERVATIVE PLACEHOLDERS chosen
# for this project, not Unitree datasheet figures. Replace them with values for
# your specific G1 revision before relying on them.
# =============================================================================
CELL_6 = r"""
import glob, numpy as np, torch, mujoco
from kimodo.assets import skeleton_asset_path
from kimodo.skeleton import G1Skeleton34, global_rots_to_local_rots
from kimodo.exports.mujoco import MujocoQposConverter, apply_g1_real_robot_projection

MAX_VEL, MAX_ACC, FPS = 10.0, 200.0, 30.0

sk = G1Skeleton34()
conv = MujocoQposConverter(sk)
model = mujoco.MjModel.from_xml_path(str(skeleton_asset_path("g1skel34", "xml", "g1.xml")))

def validate(csv_path):
    qpos = np.loadtxt(csv_path, delimiter=",")
    fails = []
    if qpos.shape[1] != model.nq:
        fails.append(f"width {qpos.shape[1]} != nq {model.nq}")
    if not np.all(np.isfinite(qpos)):
        fails.append("non-finite values")
    if abs(np.linalg.norm(qpos[:, 3:7], axis=1) - 1.0).max() > 1e-3:
        fails.append("root quaternion not unit-norm")
    cols, names, ranges = [], [], []
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE and model.jnt_limited[j]:
            cols.append(model.jnt_qposadr[j])
            names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j))
            ranges.append(model.jnt_range[j])
    for c, n, (lo, hi) in zip(cols, names, ranges):
        if (qpos[:, c] < lo).any() or (qpos[:, c] > hi).any():
            fails.append(f"{n} out of limits")
    vel = np.diff(qpos[:, cols], axis=0) * FPS
    acc = np.diff(vel, axis=0) * FPS
    data = mujoco.MjData(model)
    for frame in qpos:
        data.qpos[:] = frame
        mujoco.mj_forward(model, data)
        if not np.all(np.isfinite(data.xpos)):
            fails.append("FK produced non-finite positions")
            break
    print(f"  frames={len(qpos)} dur={len(qpos)/FPS:.2f}s "
          f"|v|max={np.abs(vel).max():.2f} |a|max={np.abs(acc).max():.1f}")
    if np.abs(vel).max() > MAX_VEL:  print(f"  WARNING: velocity over {MAX_VEL}")
    if np.abs(acc).max() > MAX_ACC:  print(f"  WARNING: acceleration over {MAX_ACC}")
    print("  RESULT:", "FAILED -> " + "; ".join(fails) if fails else "PASSED")
    return not fails

all_ok = True
for npz_path in sorted(glob.glob("/kaggle/working/outputs/*/*.npz")):
    if npz_path.endswith("_amass.npz"):
        continue
    print(npz_path)
    d = np.load(npz_path)
    pos, rot = apply_g1_real_robot_projection(
        sk, torch.from_numpy(d["posed_joints"]), torch.from_numpy(d["global_rot_mats"]),
        clamp_to_limits=True)
    local = global_rots_to_local_rots(rot, sk)
    qpos = conv.dict_to_qpos(
        {"local_rot_mats": local[None], "root_positions": pos[:, sk.root_idx, :][None]},
        torch.device("cpu"))
    out = npz_path.replace(".npz", "_clamped.csv")
    conv.save_csv(qpos, out)
    all_ok &= validate(out)

print("\nALL MOTIONS PASSED" if all_ok else "\nSOME MOTIONS FAILED")
"""

# =============================================================================
# CELL 7 - Package outputs for download.
# Kaggle cannot reach your laptop, and your laptop reaches the physical G1 on a
# local network: generation happens here, any robot step happens locally and
# separately. Commit the notebook, then pull the zip from the Output tab.
# =============================================================================
CELL_7 = r"""
!cd /kaggle/working && zip -qr kimodo_g1_motions.zip outputs && ls -la kimodo_g1_motions.zip
"""

if __name__ == "__main__":
    for i, cell in enumerate(
        [CELL_1, CELL_2, CELL_3, CELL_4, CELL_5, CELL_6, CELL_7], start=1
    ):
        print(f"\n{'=' * 70}\nCELL {i}\n{'=' * 70}{cell}")
