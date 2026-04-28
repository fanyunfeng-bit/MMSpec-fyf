#!/bin/bash
# SAGE (on EAGLE) speculative decoding evaluation for MMSpec unified dataset.
# Usage: bash scripts/Qwen2.5-VL-7B/eval_eagle_SAGE_mmspec.sh [test|testmini]

set -euo pipefail

# Configuration
BASE_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
SPEC_MODEL="${2:-Cloudriver/EAGLE-Qwen2.5-VL-7B-Instruct}"
SPLIT="${1:-test}"
DATA_FOLDER="dataset/MMSpec/${SPLIT}"
TEMPERATURE="0"
MAX_NEW_TOKEN="1024"

# EAGLE parameters
DEPTH=3
TOP_K=8
TOTAL_TOKEN=30

# SAGE parameters
SAGE_THRESHOLD_MODE="topk_ratio"   # abs | topk_ratio
SAGE_THRESHOLD_VALUE=0.1           # topk_ratio: fraction in (0,1]; abs: norm value
SAGE_MIN_SINKS=0

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$PROJECT_ROOT"

RESULT_DIR="results/Qwen2.5-VL-7B/mmspec_${SPLIT}"
ANSWER_FILE="${RESULT_DIR}/eagle_SAGE-${SAGE_THRESHOLD_MODE}-${SAGE_THRESHOLD_VALUE}-temperature-${TEMPERATURE}.jsonl"

echo "=========================================="
echo "SAGE (EAGLE) Evaluation on MMSpec (${SPLIT})"
echo "=========================================="
echo "Base Model: ${BASE_MODEL}"
echo "Spec Model: ${SPEC_MODEL}"
echo "Data Folder: ${DATA_FOLDER}"
echo "EAGLE params: depth=${DEPTH}, top_k=${TOP_K}, total_token=${TOTAL_TOKEN}"
echo "SAGE params: mode=${SAGE_THRESHOLD_MODE}, value=${SAGE_THRESHOLD_VALUE}, min_sinks=${SAGE_MIN_SINKS}"
echo "Max New Tokens: ${MAX_NEW_TOKEN}"
echo "Output: ${ANSWER_FILE}"
echo ""

mkdir -p "${RESULT_DIR}"

python -m evaluation.eval_eagle_SAGE_mmspec \
    --base-model-path "${BASE_MODEL}" \
    --spec-model-path "${SPEC_MODEL}" \
    --data-folder "${DATA_FOLDER}" \
    --answer-file "${ANSWER_FILE}" \
    --model-id "eagle_SAGE" \
    --temperature "${TEMPERATURE}" \
    --depth "${DEPTH}" \
    --top-k "${TOP_K}" \
    --total-token "${TOTAL_TOKEN}" \
    --max-new-token "${MAX_NEW_TOKEN}" \
    --sage-threshold-mode "${SAGE_THRESHOLD_MODE}" \
    --sage-threshold-value "${SAGE_THRESHOLD_VALUE}" \
    --sage-min-sinks "${SAGE_MIN_SINKS}"

echo ""
echo "SAGE (EAGLE) evaluation complete!"
echo "Results saved to: ${ANSWER_FILE}"
