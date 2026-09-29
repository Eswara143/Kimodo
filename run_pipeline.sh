#!/usr/bin/env bash
# End-to-end Kimodo-G1 pipeline: text -> LLM2Vec -> diffusion -> clamp -> CSV -> validation.
#
# Idempotent. The LoRA merge (~15 GB, several minutes) is skipped if already
# built, and re-running only regenerates motions.
#
#   ./run_pipeline.sh                          # the four default gestures
#   ./run_pipeline.sh "wave with both hands"   # one custom prompt
#
# Simulation only. Nothing here talks to a robot.

set -euo pipefail
cd "$(dirname "$0")"

# Force CPU mode globally across pipeline steps
export CUDA_VISIBLE_DEVICES=""

PY="${PY:-${PYTHON:-}}"
if [ -z "$PY" ]; then
    if [ -x "$(pwd)/venv/bin/python" ]; then
        PY="$(pwd)/venv/bin/python"
    elif [ -x "/home/sai/g1_venv/bin/python" ]; then
        PY="/home/sai/g1_venv/bin/python"
    elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
        PY="$VIRTUAL_ENV/bin/python"
    elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
        PY="$CONDA_PREFIX/bin/python"
    elif [ -x "/home/sai/miniforge3/envs/kimodo/bin/python" ]; then
        PY="/home/sai/miniforge3/envs/kimodo/bin/python"
    elif command -v conda &>/dev/null && conda info --envs | grep -q "^kimodo "; then
        PY="conda run --no-capture-output -n kimodo python"
    else
        PY="python3"
    fi
fi
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src:$(pwd)/scripts:$(pwd)/kimodo_repo"
run() { $PY "$@"; }

# Budgets sized for a 6 GB RTX 3050 with ~8-9 GB of free system RAM. The encoder
# and the diffusion model never run at the same time, so the encoder is free to
# take most of the GPU.
GPU_GIB=${GPU_GIB:-5}
CPU_GIB=${CPU_GIB:-7}

if [ "$#" -gt 0 ]; then
    PROMPTS=("$@")
    NAMES=("custom")
else
    PROMPTS=(
        "raise the right hand and wave"
        "raise the left hand and wave"
        "point forward with the right hand"
        "raise both hands above the head"
    )
    NAMES=(right_wave left_wave right_point both_raised)
fi

echo "=== 1/4  merge LLM2Vec LoRAs into base weights (once) ==="
# Folding the adapters in ahead of time keeps PEFT out of the load path, which
# is what makes disk offload -- and therefore a 15 GB encoder on a 16 GB box --
# possible at all.
run scripts/merge_llm2vec_lora.py

echo
echo "=== 2/4  encode prompts ==="
run src/precompute_text_embeddings.py "${PROMPTS[@]}" \
    --out outputs/gesture_embeddings.npz --gpu-gib "$GPU_GIB" --cpu-gib "$CPU_GIB"

echo
echo "=== 3/4  generate motions ==="
for i in "${!PROMPTS[@]}"; do
    name="${NAMES[$i]:-motion_$i}"
    echo "--- $name: ${PROMPTS[$i]} ---"
    run src/generate_g1_motion.py "${PROMPTS[$i]}" \
        --embeddings outputs/gesture_embeddings.npz \
        --output "outputs/$name" --duration 3.0 --diffusion-steps 100 --seed 42
done

echo
echo "=== 4/4  validate ==="
failed=0
for i in "${!PROMPTS[@]}"; do
    name="${NAMES[$i]:-motion_$i}"
    echo "--- $name ---"
    run src/validate_g1_motion.py "outputs/$name/$name.csv" || failed=1
done

echo
if [ "$failed" -eq 0 ]; then
    echo "ALL MOTIONS PASSED. Outputs in outputs/<gesture>/"
else
    echo "SOME MOTIONS FAILED VALIDATION - do not use them downstream."
fi
exit "$failed"
