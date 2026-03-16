#!/bin/bash
# ViSpec speculative decoding evaluation for MMSpec unified dataset
# Usage: bash scripts/Qwen/eval_vispec_mmspec.sh [test|testmini]

set -e

# Configuration - Update these paths
BASE_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
SPEC_MODEL="Cloudriver/ViSpec-Qwen2.5-VL-7B-Instruct"
SPLIT="${1:-test}"
DATA_FOLDER="dataset/MMSpec/${SPLIT}"
RESULT_NAME="qwen2.5-vl-7b"
TEMPERATURE="0"
MAX_NEW_TOKEN="1024"

# ViSpec parameters
DEPTH=3
TOP_K=8
TOTAL_TOKEN=30
NUM_Q=2

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$PROJECT_ROOT"

# Output paths
RESULT_DIR="results/Qwen2.5-VL-7B/mmspec_${SPLIT}"
ANSWER_FILE="${RESULT_DIR}/vispec-temperature-${TEMPERATURE}.jsonl"

echo "=========================================="
echo "ViSpec Evaluation on MMSpec (${SPLIT})"
echo "=========================================="
echo "Base Model: ${BASE_MODEL}"
echo "Spec Model: ${SPEC_MODEL}"
echo "Data Folder: ${DATA_FOLDER}"
echo "ViSpec params: depth=${DEPTH}, top_k=${TOP_K}, total_token=${TOTAL_TOKEN}, num_q=${NUM_Q}"
echo "Max New Tokens: ${MAX_NEW_TOKEN}"
echo "Output: ${ANSWER_FILE}"
echo ""

mkdir -p "${RESULT_DIR}"

python -m evaluation.eval_vispec_mmspec \
    --base-model-path "${BASE_MODEL}" \
    --spec-model-path "${SPEC_MODEL}" \
    --data-folder "${DATA_FOLDER}" \
    --answer-file "${ANSWER_FILE}" \
    --model-id "vispec" \
    --temperature "${TEMPERATURE}" \
    --depth "${DEPTH}" \
    --top-k "${TOP_K}" \
    --total-token "${TOTAL_TOKEN}" \
    --num-q "${NUM_Q}" \
    --max-new-token "${MAX_NEW_TOKEN}"

echo ""
echo "ViSpec evaluation complete!"
echo "Results saved to: ${ANSWER_FILE}"
