#!/bin/bash
# Baseline evaluation for MMSpec unified dataset
# Usage: bash scripts/Qwen/eval_baseline_mmspec.sh [test|testmini]

set -e

# Configuration - Update these paths
BASE_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
SPLIT="${1:-test}"  # Default to test, can pass testmini as argument
DATA_FOLDER="dataset/MMSpec/${SPLIT}"
RESULT_NAME="qwen2.5-vl-7b"
TEMPERATURE="0"
MAX_NEW_TOKEN="1024"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$PROJECT_ROOT"

# Output paths
RESULT_DIR="results/Qwen2.5-VL-7B/mmspec_${SPLIT}"
ANSWER_FILE="${RESULT_DIR}/baseline-temperature-${TEMPERATURE}.jsonl"

echo "=========================================="
echo "Baseline Evaluation on MMSpec (${SPLIT})"
echo "=========================================="
echo "Base Model: ${BASE_MODEL}"
echo "Data Folder: ${DATA_FOLDER}"
echo "Max New Tokens: ${MAX_NEW_TOKEN}"
echo "Output: ${ANSWER_FILE}"
echo ""

mkdir -p "${RESULT_DIR}"

python -m evaluation.eval_baseline_mmspec \
    --base-model-path "${BASE_MODEL}" \
    --data-folder "${DATA_FOLDER}" \
    --answer-file "${ANSWER_FILE}" \
    --model-id "baseline" \
    --temperature "${TEMPERATURE}" \
    --max-new-token "${MAX_NEW_TOKEN}"

echo ""
echo "Baseline evaluation complete!"
echo "Results saved to: ${ANSWER_FILE}"
