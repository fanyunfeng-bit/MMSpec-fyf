"""Per-query draft/verify round statistics for MMSpec results.

For each (method, query) pair, report:
  * total_rounds      -- total number of draft-generate / target-verify rounds
                         across all turns of that query
  * zero_accept_rounds -- number of those rounds in which the target accepted
                         zero draft tokens (only the bonus token was produced)
  * zero_accept_ratio -- zero_accept_rounds / total_rounds
  * total_new_tokens  -- tokens actually produced across all turns

A "round" is one (draft-propose, target-verify) cycle. In the result JSONL,
`choices[0].acceptance_length[turn_idx]` is a list whose length equals the
number of rounds used in that turn, and whose k-th element is the number of
draft tokens accepted in the k-th round (bonus token not counted). A value of
0 therefore means "draft got nothing right; target kept only its own bonus
token". For baseline (no speculative decoding, no `acceptance_length` field)
every token costs one target forward and nothing is "rejected", so
total_rounds = new_tokens and zero_accept_rounds = 0.

Usage:
    python -m evaluation.per_query_round_stats \
        --results-dir results/Qwen2.5-VL-7B/mmspec_test \
        --temperature 0

    # One CSV per method, landing next to the raw jsonl by default:
    #   results/<model>/mmspec_<split>/per_query_rounds_<method>.csv
    # The last row of each CSV is the per-query AVERAGE over that method's
    # queries (avg total_rounds, avg zero_accept_rounds, etc.).
    #
    # Use --csv-prefix to change the filename stem, --no-csv to skip writing.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

FILE_RE = re.compile(r"^(?P<name>.+)-temperature-(?P<temp>[^-]+)\.jsonl$")


def _load_jsonl(path: str) -> List[dict]:
    out = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _discover_methods(results_dir: str, temperature: str) -> List[Tuple[str, str]]:
    pattern = os.path.join(results_dir, f"*-temperature-{temperature}.jsonl")
    found = []
    for fp in sorted(glob.glob(pattern)):
        m = FILE_RE.match(os.path.basename(fp))
        if m:
            found.append((m.group("name"), fp))
    return found


def _per_query_rows(records: List[dict], method: str) -> List[dict]:
    """One aggregated row per question_id for this method."""
    rows: List[dict] = []
    for r in records:
        qid = r.get("question_id", r.get("id"))
        topic = r.get("topic", "unknown")
        category = r.get("category", "unknown")
        choices = r.get("choices") or []
        if not choices:
            continue
        c = choices[0]
        n_list = c.get("new_tokens", [])
        a_list = c.get("acceptance_length", None)
        i_list = c.get("idxs", [])

        total_new_tokens = int(sum(int(x) for x in n_list))
        num_turns = len(n_list)

        zero_available = True  # False for schemas that don't preserve per-round info
        if a_list is None:
            # Baseline / non-spec method: each token = one target forward, no
            # rejection possible.
            total_rounds = total_new_tokens
            zero_accept_rounds = 0
        else:
            total_rounds = 0
            zero_accept_rounds = 0
            for t, turn_a in enumerate(a_list):
                if isinstance(turn_a, list):
                    # EAGLE / ViSpec / Medusa / SAM / PLD / ... : per-round list
                    total_rounds += len(turn_a)
                    zero_accept_rounds += sum(1 for v in turn_a if int(v) == 0)
                elif isinstance(turn_a, (int, float)):
                    # MSD-style: only a per-turn mean survives; rounds from idxs
                    if t < len(i_list):
                        total_rounds += int(i_list[t])
                    zero_available = False
                else:
                    # Unexpected shape — skip but don't crash.
                    continue

        if a_list is not None and not zero_available:
            # Data does not let us count zero-accept rounds honestly.
            zero_accept_rounds_out = float("nan")
            zero_ratio = float("nan")
        else:
            zero_accept_rounds_out = zero_accept_rounds
            zero_ratio = (
                zero_accept_rounds / total_rounds if total_rounds > 0 else float("nan")
            )
        mat = (
            total_new_tokens / total_rounds if total_rounds > 0 else float("nan")
        )

        rows.append({
            "method": method,
            "question_id": qid,
            "topic": topic,
            "category": category,
            "num_turns": num_turns,
            "total_new_tokens": total_new_tokens,
            "total_rounds": total_rounds,
            "zero_accept_rounds": zero_accept_rounds_out,
            "zero_accept_ratio": zero_ratio,
            "mat": mat,
        })
    return rows


def _fmt(x, prec=3, nan="  -  "):
    if x is None:
        return nan
    if isinstance(x, float) and math.isnan(x):
        return nan
    if isinstance(x, float):
        return f"{x:.{prec}f}"
    return str(x)


def _print_method_summary(per_method_rows: Dict[str, List[dict]]) -> None:
    """Console summary: aggregate across all queries per method."""
    header = (
        f"{'method':<24} {'#queries':>8} {'#turns':>7} "
        f"{'#rounds':>9} {'#zero_accept':>13} {'zero_ratio':>11} {'MAT':>6}"
    )
    print(header)
    print("-" * len(header))
    for method in sorted(per_method_rows.keys()):
        rows = per_method_rows[method]
        n_q = len(rows)
        n_t = sum(r["num_turns"] for r in rows)
        n_r = sum(r["total_rounds"] for r in rows)
        z_vals = [r["zero_accept_rounds"] for r in rows]
        z_nan = any(isinstance(v, float) and math.isnan(v) for v in z_vals)
        n_z = float("nan") if z_nan else sum(int(v) for v in z_vals)
        n_tok = sum(r["total_new_tokens"] for r in rows)
        zr = float("nan") if z_nan or n_r == 0 else n_z / n_r
        mat = (n_tok / n_r) if n_r > 0 else float("nan")
        n_z_str = "N/A" if z_nan else f"{int(n_z):d}"
        print(
            f"{method:<24} {n_q:>8d} {n_t:>7d} {n_r:>9d} "
            f"{n_z_str:>13} {_fmt(zr, 4):>11} {_fmt(mat, 2):>6}"
        )


def collect(
    results_dir: str,
    temperature: str = "0",
    skip_first: int = 0,
) -> Dict[str, List[dict]]:
    methods = _discover_methods(results_dir, temperature)
    if not methods:
        raise FileNotFoundError(
            f"no files matching *-temperature-{temperature}.jsonl in {results_dir}"
        )
    per_method_rows: Dict[str, List[dict]] = {}
    for name, fp in methods:
        records = _load_jsonl(fp)[skip_first:]
        rows = _per_query_rows(records, name)
        if rows:
            per_method_rows[name] = rows
    return per_method_rows


FIELDNAMES = [
    "method", "question_id", "topic", "category",
    "num_turns", "total_new_tokens",
    "total_rounds", "zero_accept_rounds", "zero_accept_ratio", "mat",
]


def _mean(values):
    """Arithmetic mean, ignoring NaN; returns NaN on empty."""
    cleaned = [v for v in values if not (isinstance(v, float) and math.isnan(v))]
    return sum(cleaned) / len(cleaned) if cleaned else float("nan")


def _average_row(method: str, rows: List[dict]) -> dict:
    """Final AVG row: arithmetic mean over this method's queries."""
    n = len(rows)
    return {
        "method": method,
        "question_id": f"AVG({n})",
        "topic": "ALL",
        "category": "ALL",
        "num_turns":          _mean([r["num_turns"] for r in rows]),
        "total_new_tokens":   _mean([r["total_new_tokens"] for r in rows]),
        "total_rounds":       _mean([r["total_rounds"] for r in rows]),
        "zero_accept_rounds": _mean([r["zero_accept_rounds"] for r in rows]),
        "zero_accept_ratio":  _mean([r["zero_accept_ratio"] for r in rows]),
        "mat":                _mean([r["mat"] for r in rows]),
    }


def write_per_method_csvs(
    per_method_rows: Dict[str, List[dict]],
    results_dir: str,
    prefix: str = "per_query_rounds",
) -> List[str]:
    """Write one CSV per method under `results_dir`, named `<prefix>_<method>.csv`.

    The last row of each file is an AVG row with arithmetic means across that
    method's queries.
    """
    os.makedirs(results_dir, exist_ok=True)
    written: List[str] = []
    for method in sorted(per_method_rows.keys()):
        rows = sorted(per_method_rows[method], key=lambda r: r["question_id"])
        # Method name may contain chars fine for filenames in this repo, but
        # be conservative: replace path separators and whitespace.
        safe_method = re.sub(r"[^\w\-.]+", "_", method)
        path = os.path.join(results_dir, f"{prefix}_{safe_method}.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDNAMES)
            w.writeheader()
            for row in rows:
                w.writerow(row)
            w.writerow(_average_row(method, rows))
        written.append(path)
    return written


def main():
    ap = argparse.ArgumentParser(
        description="Per-query draft/verify round counts and zero-accept tallies."
    )
    ap.add_argument(
        "--results-dir",
        required=True,
        help="Directory with <method>-temperature-<t>.jsonl files "
             "(e.g. results/Qwen2.5-VL-7B/mmspec_test)",
    )
    ap.add_argument("--temperature", default="0")
    ap.add_argument(
        "--skip-first", type=int, default=0,
        help="Drop the first N records of every file (warm-up mitigation)",
    )
    group = ap.add_mutually_exclusive_group()
    group.add_argument(
        "--csv-prefix", default="per_query_rounds",
        help="Filename stem for per-method CSVs; each method writes "
             "<prefix>_<method>.csv into --results-dir. "
             "Default: per_query_rounds",
    )
    group.add_argument(
        "--no-csv", action="store_true",
        help="Do not write CSV files (only print the console summary)",
    )
    args = ap.parse_args()

    per_method_rows = collect(
        results_dir=args.results_dir,
        temperature=args.temperature,
        skip_first=args.skip_first,
    )

    _print_method_summary(per_method_rows)

    if not args.no_csv:
        paths = write_per_method_csvs(
            per_method_rows,
            results_dir=args.results_dir,
            prefix=args.csv_prefix,
        )
        print("\nPer-method CSVs written:")
        for p in paths:
            print(f"  {p}")


if __name__ == "__main__":
    main()
