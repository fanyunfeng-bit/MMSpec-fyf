#!/usr/bin/env bash
# Thin wrapper around `python -m evaluation.summarize_metrics`.
#
# By default saves summary.csv INSIDE the results directory, i.e.
#   results/<model>/mmspec_<split>/summary.csv
# so each (model, split) pair keeps its own summary alongside its raw jsonl.
#
# Usage:
#   bash scripts/summarize_metrics.sh <model> [split] [temperature] [method] [extra args...]
#
# Positional args:
#   model       e.g. Qwen2.5-VL-7B  (default)
#   split       e.g. test | testmini            (default: test)
#   temperature e.g. 0                          (default: 0)
#   method      e.g. msd  (optional; omit or use "all" for all methods)
#               When a method is given, baseline is included automatically
#               for speedup, and the default CSV is named summary_<method>.csv
#               so per-method summaries don't overwrite each other.
#
# Examples:
#   bash scripts/summarize_metrics.sh Qwen2.5-VL-7B test
#   bash scripts/summarize_metrics.sh Qwen2.5-VL-7B test 0 msd
#   bash scripts/summarize_metrics.sh Qwen2.5-VL-7B testmini 0 all --group-by topic
#   bash scripts/summarize_metrics.sh Qwen2.5-VL-7B test 0 msd --no-csv
#   bash scripts/summarize_metrics.sh Qwen2.5-VL-7B test 0 msd --csv my.csv
#   bash scripts/summarize_metrics.sh Qwen2.5-VL-7B test 0 msd --csv /tmp/out.csv

set -euo pipefail

MODEL=${1:-Qwen2.5-VL-7B}
SPLIT=${2:-test}
TEMP=${3:-0}
METHOD=${4:-}
shift $(( $# < 4 ? $# : 4 ))

RESULTS_DIR="results/${MODEL}/mmspec_${SPLIT}"

if [[ ! -d "${RESULTS_DIR}" ]]; then
    echo "error: ${RESULTS_DIR} does not exist" >&2
    exit 1
fi

# Build argument list: method filter + CSV default handling.
user_set_csv=0
disable_csv=0
extra_args=()
for arg in "$@"; do
    case "$arg" in
        --csv|--csv=*) user_set_csv=1; extra_args+=("$arg") ;;
        --no-csv)      disable_csv=1 ;;
        *)             extra_args+=("$arg") ;;
    esac
done

# Method filter: treat empty or "all" as no filter.
if [[ -n "${METHOD}" && "${METHOD}" != "all" ]]; then
    extra_args+=(--method "${METHOD}")
    default_csv="summary_${METHOD}.csv"
else
    default_csv="summary.csv"
fi

if [[ $disable_csv -eq 0 && $user_set_csv -eq 0 ]]; then
    extra_args+=(--csv "${default_csv}")
fi

python -m evaluation.summarize_metrics \
    --results-dir "${RESULTS_DIR}" \
    --temperature "${TEMP}" \
    "${extra_args[@]}"
