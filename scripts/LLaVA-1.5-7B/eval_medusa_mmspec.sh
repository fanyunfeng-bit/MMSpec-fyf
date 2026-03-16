#!/bin/bash
# Medusa speculative decoding evaluation for MMSpec unified dataset
# Usage: bash scripts/LLaVA-1.5-7B/eval_medusa_mmspec.sh [test|testmini]

set -euo pipefail

BASE_MODEL="llava-hf/llava-1.5-7b-hf"
SPEC_MODEL="checkpoints_medusa_llava/state_19"
SPLIT="${1:-test}"
DATA_FOLDER="dataset/MMSpec/${SPLIT}"
TEMPERATURE="0"
MAX_NEW_TOKEN="1024"

DEPTH=3
TOP_K=8
TOTAL_TOKEN=30

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$PROJECT_ROOT"

RESULT_DIR="results/LLaVA-1.5-7B/mmspec_${SPLIT}"
ANSWER_FILE="${RESULT_DIR}/medusa-temperature-${TEMPERATURE}.jsonl"

echo "=========================================="
echo "Medusa Evaluation on MMSpec (${SPLIT})"
echo "=========================================="
echo "Base Model: ${BASE_MODEL}"
echo "Spec Model: ${SPEC_MODEL}"
echo "Data Folder: ${DATA_FOLDER}"
echo "Medusa params: depth=${DEPTH}, top_k=${TOP_K}, total_token=${TOTAL_TOKEN}"
echo "Max New Tokens: ${MAX_NEW_TOKEN}"
echo "Output: ${ANSWER_FILE}"
echo ""

mkdir -p "${RESULT_DIR}"

python -m evaluation.eval_medusa_mmspec \
    --base-model-path "${BASE_MODEL}" \
    --spec-model-path "${SPEC_MODEL}" \
    --data-folder "${DATA_FOLDER}" \
    --answer-file "${ANSWER_FILE}" \
    --model-id "medusa" \
    --temperature "${TEMPERATURE}" \
    --depth "${DEPTH}" \
    --top-k "${TOP_K}" \
    --total-token "${TOTAL_TOKEN}" \
    --max-new-token "${MAX_NEW_TOKEN}"

echo ""
echo "Medusa evaluation complete!"
echo "Results saved to: ${ANSWER_FILE}"
