#!/bin/bash
# Token Recycling evaluation for MMSpec unified dataset
# Usage: bash scripts/LLaVA-1.5-7B/eval_recycling_mmspec.sh [test|testmini]

set -euo pipefail

BASE_MODEL="llava-hf/llava-1.5-7b-hf"
SPEC_MODEL=""
SPLIT="${1:-test}"
DATA_FOLDER="dataset/MMSpec/${SPLIT}"
TEMPERATURE="0"
MAX_NEW_TOKEN="1024"

MATRIX_TOP_K=8
DRAFT_LEN=10

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$PROJECT_ROOT"

RESULT_DIR="results/LLaVA-1.5-7B/mmspec_${SPLIT}"
ANSWER_FILE="${RESULT_DIR}/recycling-temperature-${TEMPERATURE}.jsonl"

echo "=========================================="
echo "Token Recycling Evaluation on MMSpec (${SPLIT})"
echo "=========================================="
echo "Base Model: ${BASE_MODEL}"
echo "Data Folder: ${DATA_FOLDER}"
echo "Recycling params: matrix_top_k=${MATRIX_TOP_K}, draft_len=${DRAFT_LEN}"
echo "Max New Tokens: ${MAX_NEW_TOKEN}"
echo "Output: ${ANSWER_FILE}"
echo ""

mkdir -p "${RESULT_DIR}"

python -m evaluation.eval_recycling_mmspec \
    --base-model-path "${BASE_MODEL}" \
    --spec-model-path "${SPEC_MODEL}" \
    --data-folder "${DATA_FOLDER}" \
    --answer-file "${ANSWER_FILE}" \
    --model-id "recycling" \
    --temperature "${TEMPERATURE}" \
    --max-new-token "${MAX_NEW_TOKEN}" \
    --matrix-top-k "${MATRIX_TOP_K}" \
    --draft-len "${DRAFT_LEN}"

echo ""
echo "Token Recycling evaluation complete!"
echo "Results saved to: ${ANSWER_FILE}"
