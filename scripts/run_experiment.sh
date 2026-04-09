#!/bin/bash
# Run a full experiment pipeline:
# 1. Generate trajectories (or use pre-generated)
# 2. Score and select trajectories
# 3. Train student model
# 4. Evaluate

set -euo pipefail

# === Configuration ===
CONDITION=${1:?Usage: run_experiment.sh <condition> <student> [seed]}
STUDENT=${2:?Usage: run_experiment.sh <condition> <student> [seed]}
SEED=${3:-42}

# Paths
CODE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${CODE_DIR}/data/processed"
TRAJ_DIR="${CODE_DIR}/data/trajectories"
OUTPUT_BASE="${CODE_DIR}/outputs"

# Model names
declare -A STUDENT_MODELS=(
    ["qwen3-8b"]="Qwen/Qwen3-8B"
    ["qwen3-4b"]="Qwen/Qwen3-4B"
)
STUDENT_MODEL="${STUDENT_MODELS[$STUDENT]}"
OUTPUT_DIR="${OUTPUT_BASE}/sft/${CONDITION}_${STUDENT}_seed${SEED}"

echo "=========================================="
echo "Experiment: ${CONDITION} / ${STUDENT} / seed=${SEED}"
echo "Output: ${OUTPUT_DIR}"
echo "=========================================="

# === Step 1: Trajectory Selection ===
echo "[Step 1] Selecting trajectories with method: ${CONDITION}"
SELECTED_TRAJ="${OUTPUT_DIR}/selected_trajectories.jsonl"
mkdir -p "${OUTPUT_DIR}"

python "${CODE_DIR}/selection/select.py" \
    --method "${CONDITION}" \
    --trajectories "${TRAJ_DIR}/all_trajectories.jsonl" \
    --student_model "${STUDENT_MODEL}" \
    --output "${SELECTED_TRAJ}" \
    --seed "${SEED}" \
    --k 500

# === Step 2: SFT Training ===
echo "[Step 2] Training ${STUDENT} on selected trajectories"
python "${CODE_DIR}/training/sft_trainer.py" \
    --model "${STUDENT_MODEL}" \
    --train_data "${SELECTED_TRAJ}" \
    --output_dir "${OUTPUT_DIR}/checkpoint" \
    --seed "${SEED}" \
    --batch_size 4 \
    --grad_accum 4 \
    --epochs 3 \
    --wandb_project "search_traj_selection" \
    --wandb_run_name "${CONDITION}_${STUDENT}_s${SEED}"

# === Step 3: Evaluation ===
echo "[Step 3] Evaluating on multi-hop QA benchmarks"
python "${CODE_DIR}/evaluation/evaluate.py" \
    --model_path "${OUTPUT_DIR}/checkpoint" \
    --datasets \
        "${DATA_DIR}/hotpotqa_dev.jsonl" \
        "${DATA_DIR}/musique_dev.jsonl" \
        "${DATA_DIR}/2wikimhqa_dev.jsonl" \
    --output "${OUTPUT_DIR}/eval_results.json"

echo "=========================================="
echo "Experiment complete! Results: ${OUTPUT_DIR}/eval_results.json"
echo "=========================================="
