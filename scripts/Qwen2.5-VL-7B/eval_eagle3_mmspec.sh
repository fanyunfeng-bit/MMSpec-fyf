#!/bin/bash
# EAGLE3 evaluation on MMSpec using existing EAGLE evaluation framework
# Usage:
#   bash scripts/Qwen2.5-VL-7B/eval_eagle3_mmspec.sh [test|testmini] [spec_model_dir(optional)]

set -euo pipefail

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
SPLIT="${1:-test}"
SPEC_MODEL="${2:-Cloudriver/EAGLE3-Qwen2.5-VL-7B-Instruct}"

TEMPERATURE="${TEMPERATURE:-0}"
MAX_NEW_TOKEN="${MAX_NEW_TOKEN:-1024}"
DEPTH="${DEPTH:-3}"
TOP_K="${TOP_K:-8}"
TOTAL_TOKEN="${TOTAL_TOKEN:-30}"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$PROJECT_ROOT"



DATA_FOLDER="dataset/MMSpec/${SPLIT}"
RESULT_DIR="results/Qwen2.5-VL-7B/mmspec_${SPLIT}"
ANSWER_FILE="${RESULT_DIR}/eagle3-temperature-${TEMPERATURE}.jsonl"

echo "=========================================="
echo "EAGLE3 Evaluation on MMSpec (${SPLIT})"
echo "=========================================="
echo "Base Model: ${BASE_MODEL}"
echo "Spec Model: ${SPEC_MODEL}"
echo "Data Folder: ${DATA_FOLDER}"
echo "EAGLE params: depth=${DEPTH}, top_k=${TOP_K}, total_token=${TOTAL_TOKEN}"
echo "Max New Tokens: ${MAX_NEW_TOKEN}"
echo "Output: ${ANSWER_FILE}"
echo ""

mkdir -p "${RESULT_DIR}"

python3 -m evaluation.eval_eagle3_mmspec \
    --base-model-path "${BASE_MODEL}" \
    --spec-model-path "${SPEC_MODEL}" \
    --data-folder "${DATA_FOLDER}" \
    --answer-file "${ANSWER_FILE}" \
    --model-id "eagle3" \
    --temperature "${TEMPERATURE}" \
    --depth "${DEPTH}" \
    --top-k "${TOP_K}" \
    --total-token "${TOTAL_TOKEN}" \
    --max-new-token "${MAX_NEW_TOKEN}"

echo ""
echo "EAGLE3 evaluation complete!"
echo "Results saved to: ${ANSWER_FILE}"
