#!/bin/bash
# Lookahead speculative decoding evaluation for MMSpec unified dataset
# Usage: bash scripts/Qwen/eval_lookahead_mmspec.sh [test|testmini]

set -e

# Configuration - Update these paths
BASE_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
SPEC_MODEL=""  # Lookahead doesn't require a separate spec model
SPLIT="${1:-test}"
DATA_FOLDER="dataset/MMSpec/${SPLIT}"
RESULT_NAME="qwen2.5-vl-7b"
TEMPERATURE="0"
MAX_NEW_TOKEN="1024"

# Lookahead parameters
DECODING_LENGTH=64
BRANCH_LENGTH=12

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$PROJECT_ROOT"

# Output paths
RESULT_DIR="results/Qwen2.5-VL-7B/mmspec_${SPLIT}"
ANSWER_FILE="${RESULT_DIR}/lookahead-temperature-${TEMPERATURE}.jsonl"

echo "=========================================="
echo "Lookahead Evaluation on MMSpec (${SPLIT})"
echo "=========================================="
echo "Base Model: ${BASE_MODEL}"
echo "Data Folder: ${DATA_FOLDER}"
echo "Lookahead params: decoding_length=${DECODING_LENGTH}, branch_length=${BRANCH_LENGTH}"
echo "Max New Tokens: ${MAX_NEW_TOKEN}"
echo "Output: ${ANSWER_FILE}"
echo ""

mkdir -p "${RESULT_DIR}"

python -m evaluation.eval_lookahead_mmspec \
    --base-model-path "${BASE_MODEL}" \
    --spec-model-path "${SPEC_MODEL}" \
    --data-folder "${DATA_FOLDER}" \
    --answer-file "${ANSWER_FILE}" \
    --model-id "lookahead" \
    --temperature "${TEMPERATURE}" \
    --max-new-token "${MAX_NEW_TOKEN}" \
    --decoding-length "${DECODING_LENGTH}" \
    --branch-length "${BRANCH_LENGTH}"

echo ""
echo "Lookahead evaluation complete!"
echo "Results saved to: ${ANSWER_FILE}"
