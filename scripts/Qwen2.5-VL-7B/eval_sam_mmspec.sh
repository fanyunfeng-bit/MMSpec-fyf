#!/bin/bash
# SAM speculative decoding evaluation for MMSpec unified dataset
# Usage: bash scripts/Qwen/eval_sam_mmspec.sh [test|testmini]

set -e

# Configuration - Update these paths
BASE_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
SPEC_MODEL=""  # SAM doesn't require a separate spec model
SPLIT="${1:-test}"
DATA_FOLDER="dataset/MMSpec/${SPLIT}"
RESULT_NAME="qwen2.5-vl-7b"
TEMPERATURE="0"
MAX_NEW_TOKEN="1024"

# SAM parameters
TOTAL_TOKEN=30
DEPTH=3
TOP_K=8
THRESHOLD=1.0

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$PROJECT_ROOT"

# Output paths
RESULT_DIR="results/Qwen2.5-VL-7B/mmspec_${SPLIT}"
ANSWER_FILE="${RESULT_DIR}/sam-temperature-${TEMPERATURE}.jsonl"

echo "=========================================="
echo "SAM Evaluation on MMSpec (${SPLIT})"
echo "=========================================="
echo "Base Model: ${BASE_MODEL}"
echo "Spec Model: ${SPEC_MODEL}"
echo "Data Folder: ${DATA_FOLDER}"
echo "SAM params: total_token=${TOTAL_TOKEN}, depth=${DEPTH}, top_k=${TOP_K}, threshold=${THRESHOLD}"
echo "Max New Tokens: ${MAX_NEW_TOKEN}"
echo "Output: ${ANSWER_FILE}"
echo ""

mkdir -p "${RESULT_DIR}"

python -m evaluation.eval_sam_mmspec \
    --base-model-path "${BASE_MODEL}" \
    --spec-model-path "${SPEC_MODEL}" \
    --data-folder "${DATA_FOLDER}" \
    --answer-file "${ANSWER_FILE}" \
    --model-id "sam" \
    --temperature "${TEMPERATURE}" \
    --max-new-token "${MAX_NEW_TOKEN}" \
    --total-token "${TOTAL_TOKEN}" \
    --depth "${DEPTH}" \
    --top-k "${TOP_K}" \
    --threshold "${THRESHOLD}"

echo ""
echo "SAM evaluation complete!"
echo "Results saved to: ${ANSWER_FILE}"
