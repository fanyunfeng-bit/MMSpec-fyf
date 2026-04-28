"""Aggregate MAT and Walltime Speedup for MMSpec result files.

Reads per-method JSONL files under a results directory shaped like:
    results/<model>/mmspec_<split>/<method>-temperature-<t>.jsonl
and prints per-method Mean Accepted Tokens (MAT) and speedup versus a
baseline run. Optional CSV export and per-topic breakdown.

Usage:
    python -m evaluation.summarize_metrics \
        --results-dir results/Qwen2.5-VL-7B/mmspec_test \
        --baseline baseline --temperature 0

No existing file is modified; this module is self-contained.
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
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _iter_turns(records: Iterable[dict]):
    """Yield one dict per (sample, turn) with new_tokens, wall_time, accept,
    idx, draft_time, target_time. `accept` keeps whatever raw shape the
    evaluator stored (list | scalar | None) — downstream helpers dispatch."""
    for r in records:
        qid = r.get("question_id", r.get("id"))
        topic = r.get("topic", "unknown")
        category = r.get("category", "unknown")
        choices = r.get("choices") or []
        if not choices:
            continue
        c = choices[0]
        n_list = c.get("new_tokens", [])
        w_list = c.get("wall_time", [])
        a_list = c.get("acceptance_length", [])
        i_list = c.get("idxs", [])
        d_list = c.get("draft_time", [])
        t_list = c.get("target_time", [])
        for t in range(len(n_list)):
            yield {
                "qid": qid,
                "topic": topic,
                "category": category,
                "turn": t,
                "new_tokens": int(n_list[t]) if t < len(n_list) else 0,
                "wall_time": float(w_list[t]) if t < len(w_list) else 0.0,
                "accept": a_list[t] if t < len(a_list) else None,
                "idx": int(i_list[t]) if t < len(i_list) else 0,
                "draft_time": float(d_list[t]) if t < len(d_list) else 0.0,
                "target_time": float(t_list[t]) if t < len(t_list) else 0.0,
            }


def _num_verify_rounds(row: dict) -> int:
    """Rounds of (draft-propose, target-verify) used in this turn.

    Three schemas observed in the repo:
      * list[int]  -- EAGLE / ViSpec / Medusa / SAM / PLD / ... : len() is rounds
      * scalar     -- MSD stores only a per-turn mean; rounds come from `idxs[t]`
      * None       -- baseline (no spec); every new token costs one target forward
    """
    al = row["accept"]
    if isinstance(al, list):
        return len(al)
    if isinstance(al, (int, float)):
        return max(int(row.get("idx", 0)), 0)
    return max(row["new_tokens"], 0)


def compute_mat(rows: List[dict]) -> float:
    """MAT = total new_tokens / total verify rounds."""
    total_tok = 0
    total_rounds = 0
    for row in rows:
        total_tok += row["new_tokens"]
        total_rounds += _num_verify_rounds(row)
    return total_tok / total_rounds if total_rounds else float("nan")


def compute_draft_accept_mean(rows: List[dict]) -> float:
    """Mean of recorded acceptance_length values (draft-accepted only, no bonus).

    Handles both schemas:
      * list[int]  -- flat mean across all recorded per-round values
      * scalar     -- per-turn already-averaged value, weighted by that turn's
                      number of rounds (so all methods are comparable)
    Returns NaN for baseline / rows without the field."""
    total, weight = 0.0, 0.0
    for row in rows:
        al = row["accept"]
        if isinstance(al, list):
            for v in al:
                total += float(v)
                weight += 1
        elif isinstance(al, (int, float)):
            n_rounds = _num_verify_rounds(row)
            if n_rounds > 0:
                total += float(al) * n_rounds
                weight += n_rounds
    return (total / weight) if weight > 0 else float("nan")


def compute_speedup(
    method_rows: List[dict],
    baseline_rows: List[dict],
    mode: str = "paired",
) -> float:
    """mode='paired': per-turn paired mean ratio (EAGLE-paper style).
       mode='global': ratio of aggregate throughput (tokens/sec)."""
    if mode == "paired":
        base = {(r["qid"], r["turn"]): r for r in baseline_rows}
        ratios = []
        for r in method_rows:
            b = base.get((r["qid"], r["turn"]))
            if b is None:
                continue
            if r["wall_time"] <= 0 or b["wall_time"] <= 0 or b["new_tokens"] <= 0:
                continue
            tp_m = r["new_tokens"] / r["wall_time"]
            tp_b = b["new_tokens"] / b["wall_time"]
            if tp_b > 0:
                ratios.append(tp_m / tp_b)
        return sum(ratios) / len(ratios) if ratios else float("nan")
    elif mode == "global":
        def tp(rs):
            N = sum(r["new_tokens"] for r in rs)
            T = sum(r["wall_time"] for r in rs)
            return (N / T) if T > 0 else 0.0
        tp_m, tp_b = tp(method_rows), tp(baseline_rows)
        return (tp_m / tp_b) if tp_b > 0 else float("nan")
    else:
        raise ValueError(f"unknown speedup mode: {mode}")


def _discover_methods(results_dir: str, temperature: str) -> List[Tuple[str, str]]:
    out = []
    pattern = os.path.join(results_dir, f"*-temperature-{temperature}.jsonl")
    for fp in sorted(glob.glob(pattern)):
        m = FILE_RE.match(os.path.basename(fp))
        if m:
            out.append((m.group("name"), fp))
    return out


def _fmt(x: float, prec: int = 3, nan: str = "  -  ") -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return nan
    return f"{x:.{prec}f}"


def summarize(
    results_dir: str,
    baseline_name: str = "baseline",
    temperature: str = "0",
    group_by: Optional[str] = None,
    skip_first: int = 0,
    csv_path: Optional[str] = None,
    methods_filter: Optional[List[str]] = None,
) -> List[dict]:
    methods = _discover_methods(results_dir, temperature)
    if not methods:
        raise FileNotFoundError(
            f"no files matching *-temperature-{temperature}.jsonl in {results_dir}"
        )

    if methods_filter:
        wanted = set(methods_filter)
        # Always keep baseline around so speedup can still be computed, even
        # if the user didn't list it.
        wanted.add(baseline_name)
        available = {n for n, _ in methods}
        missing = [m for m in methods_filter if m not in available]
        if missing:
            raise FileNotFoundError(
                f"method(s) not found in {results_dir} at temperature={temperature}: "
                f"{missing}. Available: {sorted(available)}"
            )
        methods = [(n, fp) for n, fp in methods if n in wanted]

    baseline_path = os.path.join(
        results_dir, f"{baseline_name}-temperature-{temperature}.jsonl"
    )
    have_baseline = os.path.exists(baseline_path)
    baseline_rows: List[dict] = []
    if have_baseline:
        base_records = _load_jsonl(baseline_path)[skip_first:]
        baseline_rows = list(_iter_turns(base_records))

    # Grouping keys: None (overall), 'topic', or 'category'
    def key_of(row):
        if group_by is None:
            return "ALL"
        return row.get(group_by, "unknown")

    results: List[dict] = []
    for name, fp in methods:
        records = _load_jsonl(fp)[skip_first:]
        rows = list(_iter_turns(records))
        if not rows:
            continue

        # Split rows by group key.
        groups: Dict[str, List[dict]] = defaultdict(list)
        base_groups: Dict[str, List[dict]] = defaultdict(list)
        for r in rows:
            groups[key_of(r)].append(r)
        for r in baseline_rows:
            base_groups[key_of(r)].append(r)

        for g, g_rows in groups.items():
            entry = {
                "method": name,
                "group": g,
                "n_samples": len({r["qid"] for r in g_rows}),
                "n_turns": len(g_rows),
                "total_new_tokens": sum(r["new_tokens"] for r in g_rows),
                "total_wall_time": sum(r["wall_time"] for r in g_rows),
                "total_draft_time": sum(r["draft_time"] for r in g_rows),
                "total_target_time": sum(r["target_time"] for r in g_rows),
                "MAT": compute_mat(g_rows),
                "draft_accept_mean": compute_draft_accept_mean(g_rows),
            }
            if have_baseline and name != baseline_name:
                b_rows = base_groups.get(g, [])
                entry["speedup_paired"] = compute_speedup(g_rows, b_rows, "paired")
                entry["speedup_global"] = compute_speedup(g_rows, b_rows, "global")
            else:
                entry["speedup_paired"] = 1.0 if name == baseline_name else float("nan")
                entry["speedup_global"] = 1.0 if name == baseline_name else float("nan")
            results.append(entry)

    # Pretty print.
    header = f"{'method':<24} {'group':<28} {'#samp':>6} {'#turn':>6} " \
             f"{'MAT':>6} {'draft_acc':>10} {'Speed(p)':>9} {'Speed(g)':>9}"
    print(header)
    print("-" * len(header))
    # Put baseline first, then methods sorted alphabetically; within method
    # keep group order as discovered.
    def sort_key(e):
        is_base = 0 if e["method"] == baseline_name else 1
        return (is_base, e["method"], e["group"])

    for e in sorted(results, key=sort_key):
        print(
            f"{e['method']:<24} {str(e['group'])[:28]:<28} "
            f"{e['n_samples']:>6d} {e['n_turns']:>6d} "
            f"{_fmt(e['MAT'], 2):>6} {_fmt(e['draft_accept_mean'], 2):>10} "
            f"{_fmt(e['speedup_paired'], 3):>9} {_fmt(e['speedup_global'], 3):>9}"
        )

    if csv_path:
        # Bare filename (no directory component) → place inside the results
        # directory, so summaries live next to the files they came from:
        #   results/<model>/mmspec_<split>/<csv>
        if not os.path.isabs(csv_path) and os.path.dirname(csv_path) == "":
            csv_path = os.path.join(results_dir, csv_path)
        fieldnames = [
            "method", "group", "n_samples", "n_turns",
            "total_new_tokens", "total_wall_time",
            "total_draft_time", "total_target_time",
            "MAT", "draft_accept_mean",
            "speedup_paired", "speedup_global",
        ]
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for e in sorted(results, key=sort_key):
                w.writerow(e)
        print(f"\nCSV written to: {csv_path}")

    return results


def main():
    ap = argparse.ArgumentParser(
        description="Summarize MAT and walltime speedup for MMSpec results."
    )
    ap.add_argument(
        "--results-dir",
        required=True,
        help="Directory containing <method>-temperature-<t>.jsonl files "
             "(e.g. results/Qwen2.5-VL-7B/mmspec_test)",
    )
    ap.add_argument(
        "--baseline",
        default="baseline",
        help="Method name to use as baseline for speedup (default: baseline)",
    )
    ap.add_argument(
        "--temperature",
        default="0",
        help="Temperature suffix to select files (default: 0)",
    )
    ap.add_argument(
        "--group-by",
        choices=["topic", "category"],
        default=None,
        help="Break down metrics by topic or category (default: overall only)",
    )
    ap.add_argument(
        "--skip-first",
        type=int,
        default=0,
        help="Drop the first N records of every file (warm-up mitigation)",
    )
    ap.add_argument(
        "--csv",
        default=None,
        help="Also write a CSV with all aggregated rows to this path",
    )
    ap.add_argument(
        "--method",
        nargs="+",
        default=None,
        help="Only summarize the given method(s), e.g. `--method msd` or "
             "`--method msd eagle vispec`. Baseline is always included "
             "(needed for speedup).",
    )
    args = ap.parse_args()

    summarize(
        results_dir=args.results_dir,
        baseline_name=args.baseline,
        temperature=args.temperature,
        group_by=args.group_by,
        skip_first=args.skip_first,
        csv_path=args.csv,
        methods_filter=args.method,
    )


if __name__ == "__main__":
    main()
