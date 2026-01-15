#!/usr/bin/env bash
set -euo pipefail

# Run partial-observability ablation for value-based methods.
# Matrix: obs_type {pixel,ram} x frame_stack {1,4}
# Algorithms: dqn, iqn
#
# Examples:
#   bash scripts/run_pomdp_ablation.sh
#   bash scripts/run_pomdp_ablation.sh ALE/Pong-v5 ALE/Breakout-v5
#
# Notes:
# - The experiment name prefix is set to encode (alg, obs, fs, seed).
# - The trainer already appends env name + timestamp, so runs won't overwrite.

ENVS=("$@")
if [ ${#ENVS[@]} -eq 0 ]; then
  ENVS=("ALE/Pong-v5" "ALE/Breakout-v5" "ALE/VideoPinball-v5")
fi

# You can edit these seeds or export SEEDS="0 1 2" before running.
SEEDS_STR=${SEEDS:-"42 43 44"}
read -r -a SEEDS <<< "${SEEDS_STR}"

CONFIGS=(
  "configs/ablation/value_based_pixel_fs1.yaml"
  "configs/ablation/value_based_pixel_fs4.yaml"
  "configs/ablation/value_based_ram_fs1.yaml"
  "configs/ablation/value_based_ram_fs4.yaml"
)

ALGS=("dqn" "iqn")

for env in "${ENVS[@]}"; do
  for alg in "${ALGS[@]}"; do
    for cfg in "${CONFIGS[@]}"; do
      base=$(basename "${cfg}" .yaml)  # e.g. value_based_ram_fs4
      for seed in "${SEEDS[@]}"; do
        exp="pomdp-${alg}-${base}-seed${seed}"
        echo "[RUN] env=${env} alg=${alg} cfg=${cfg} seed=${seed} exp=${exp}"
        python train.py --algorithm "${alg}" --config "${cfg}" --env "${env}" --seed "${seed}" --exp-name "${exp}"
      done
    done
  done
done
