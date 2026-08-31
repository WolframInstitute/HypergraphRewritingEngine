#!/usr/bin/env python3
"""Compare two benchmark_suite summary.csv files metric by metric and fail on a regression.

INPUT FORMAT. `BenchmarkRegistry::write_summary_csv` (benchmarking/benchmark_framework.cpp)
writes exactly two lines: a header and one row. The first four columns are
`hash_type,hash,commit_date,timestamp`; every column after them is
`<benchmark>_<param values>_<stat>` from `make_column_name`, with the five stats written in the
order `avg_us, stddev_us, samples, min_us, max_us`. A metric here is the `<benchmark>_<params>`
prefix, and the gated statistic is one of its timing columns (`--stat`, default `avg_us`).

DIRECTION. The summary carries no result type, so it is read from the `detailed.csv` beside each
summary (`result_type` column, written by `write_detailed_csv`). A benchmark whose type is
`RATIO` (the `comparative_*_speedup` family) reports a speedup, and there a FALL is the
regression; every other metric is a time, where a RISE is. A metric is attributed to the
benchmark whose name is its longest `<name>_` prefix, which is what separates
`comparative_config1` from `comparative_config1_speedup`. Without a `detailed.csv` every metric
is treated as a time.

TOLERANCE. A metric regresses when its worsening -- the fractional rise of a time, or the
fractional fall of a ratio -- exceeds `--tolerance` (default 0.10). A metric present in only one
file is reported as `new` or `missing` and never fails the comparison: a filtered run measures a
subset, and a renamed benchmark is a change to the suite, not to the engine.

Exit status: 0 when no metric regresses, 1 when at least one does, 2 on a malformed input.

Usage:
    perf_compare.py BASELINE_SUMMARY HEAD_SUMMARY [--tolerance F] [--stat S] [--limit N]
    perf_compare.py --list SUMMARY          print every parsed metric with its five stats
    perf_compare.py --self-test             synthetic summaries in a temp dir; exit 0 on pass
"""

import argparse
import csv
import math
import os
import sys
import tempfile
from collections import namedtuple

META_COLUMNS = ("hash_type", "hash", "commit_date", "timestamp")
STATS = ("avg_us", "stddev_us", "samples", "min_us", "max_us")
TIMING_STATS = ("avg_us", "min_us", "max_us")
DEFAULT_TOLERANCE = 0.10

Row = namedtuple("Row", "metric status worsening before after n_before n_after")


class SummaryError(Exception):
    pass


def parse_summary(path):
    """Return (meta, metrics): meta maps the four leading columns; metrics maps
    `<benchmark>_<params>` to a dict over STATS."""
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if len(rows) != 2:
        raise SummaryError(f"{path}: expected a header and one row, found {len(rows)} lines")
    header, values = rows
    if len(header) != len(values):
        raise SummaryError(f"{path}: header has {len(header)} columns, row has {len(values)}")
    if tuple(header[: len(META_COLUMNS)]) != META_COLUMNS:
        raise SummaryError(f"{path}: leading columns are {header[:4]}, expected {list(META_COLUMNS)}")
    meta = dict(zip(META_COLUMNS, values[: len(META_COLUMNS)]))
    metrics = {}
    for column, value in zip(header[len(META_COLUMNS):], values[len(META_COLUMNS):]):
        for stat in STATS:
            suffix = "_" + stat
            if column.endswith(suffix):
                metric = column[: -len(suffix)]
                break
        else:
            raise SummaryError(f"{path}: column {column!r} ends in none of {STATS}")
        if not metric:
            raise SummaryError(f"{path}: column {column!r} has no metric name")
        try:
            metrics.setdefault(metric, {})[stat] = float(value)
        except ValueError:
            raise SummaryError(f"{path}: column {column!r} holds {value!r}, not a number")
    for metric, stats in metrics.items():
        missing = [s for s in STATS if s not in stats]
        if missing:
            raise SummaryError(f"{path}: metric {metric!r} lacks {missing}")
    return meta, metrics


def result_types(summary_path):
    """Map benchmark_name -> result_type from the detailed.csv beside a summary, when present."""
    detailed = os.path.join(os.path.dirname(os.path.abspath(summary_path)), "detailed.csv")
    types = {}
    if not os.path.exists(detailed):
        return types
    with open(detailed, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "result_type" not in reader.fieldnames:
            return types
        for row in reader:
            types[row["benchmark_name"]] = row["result_type"]
    return types


def is_ratio(metric, types):
    best = None
    for name, result_type in types.items():
        if metric == name or metric.startswith(name + "_"):
            if best is None or len(name) > len(best):
                best = name
    return best is not None and types[best] == "RATIO"


def fractional_change(before, after):
    if before == 0.0:
        return 0.0 if after == 0.0 else math.inf
    return (after - before) / before


def compare(base_metrics, head_metrics, types, tolerance, stat):
    rows = []
    for metric in sorted(set(base_metrics) | set(head_metrics)):
        b = base_metrics.get(metric)
        h = head_metrics.get(metric)
        if b is None:
            rows.append(Row(metric, "new", None, None, h[stat], None, h["samples"]))
            continue
        if h is None:
            rows.append(Row(metric, "missing", None, b[stat], None, b["samples"], None))
            continue
        change = fractional_change(b[stat], h[stat])
        worsening = -change if is_ratio(metric, types) else change
        if worsening > tolerance:
            status = "REGRESSION"
        elif worsening < -tolerance:
            status = "improved"
        else:
            status = "ok"
        rows.append(Row(metric, status, worsening, b[stat], h[stat], b["samples"], h["samples"]))

    def key(row):
        if row.worsening is None:
            return (1, 0.0, row.metric)
        return (0, -row.worsening, row.metric)

    rows.sort(key=key)
    return rows


def fmt_value(v):
    return "-" if v is None else f"{v:.3f}"


def fmt_samples(v):
    return "-" if v is None else f"{int(v)}"


def fmt_worsening(w):
    if w is None:
        return "-"
    if math.isinf(w):
        return "+inf" if w > 0 else "-inf"
    return f"{w * 100:+.1f}%"


def render(rows, base_path, head_path, base_meta, head_meta, tolerance, stat, limit, out):
    regressions = [r for r in rows if r.status == "REGRESSION"]
    counts = {}
    for r in rows:
        counts[r.status] = counts.get(r.status, 0) + 1
    print(f"baseline: {base_path} ({base_meta['hash_type']} {base_meta['hash']}, {base_meta['timestamp']})", file=out)
    print(f"head:     {head_path} ({head_meta['hash_type']} {head_meta['hash']}, {head_meta['timestamp']})", file=out)
    print(f"stat: {stat}   tolerance: {tolerance:.3f}   metrics: {len(rows)}   "
          + "   ".join(f"{k}: {v}" for k, v in sorted(counts.items())), file=out)
    print(f"{'STATUS':<10} {'WORSE':>8} {'BEFORE':>14} {'AFTER':>14} {'N':>9}  METRIC", file=out)
    shown = rows if limit is None else rows[:limit]
    for r in shown:
        n = f"{fmt_samples(r.n_before)}/{fmt_samples(r.n_after)}"
        print(f"{r.status:<10} {fmt_worsening(r.worsening):>8} {fmt_value(r.before):>14} "
              f"{fmt_value(r.after):>14} {n:>9}  {r.metric}", file=out)
    if limit is not None and len(rows) > limit:
        print(f"... {len(rows) - limit} more rows (--limit {limit})", file=out)
    if regressions:
        print(f"FAIL: {len(regressions)} metric(s) worsened by more than {tolerance * 100:.1f}%:", file=out)
        for r in regressions:
            print(f"  {r.metric}: {fmt_value(r.before)} -> {fmt_value(r.after)} ({fmt_worsening(r.worsening)})",
                  file=out)
    else:
        print(f"PASS: no metric worsened by more than {tolerance * 100:.1f}%", file=out)
    return 1 if regressions else 0


def run_compare(base_path, head_path, tolerance, stat, limit, out=sys.stdout):
    base_meta, base_metrics = parse_summary(base_path)
    head_meta, head_metrics = parse_summary(head_path)
    types = dict(result_types(base_path))
    types.update(result_types(head_path))
    rows = compare(base_metrics, head_metrics, types, tolerance, stat)
    code = render(rows, base_path, head_path, base_meta, head_meta, tolerance, stat, limit, out)
    return code, rows


def list_metrics(path, out=sys.stdout):
    meta, metrics = parse_summary(path)
    types = result_types(path)
    print(f"{path}: {meta['hash_type']} {meta['hash']} commit_date={meta['commit_date']} "
          f"timestamp={meta['timestamp']} metrics={len(metrics)} "
          f"ratio_metrics={sum(1 for m in metrics if is_ratio(m, types))}", file=out)
    print(f"{'METRIC':<70} {'TYPE':<5} " + " ".join(f"{s:>14}" for s in STATS), file=out)
    for metric, stats in metrics.items():
        kind = "RATIO" if is_ratio(metric, types) else "TIME"
        print(f"{metric:<70} {kind:<5} " + " ".join(f"{stats[s]:>14.6f}" for s in STATS), file=out)


# ---------------------------------------------------------------------------------------------
# Self-test: synthetic summaries in the suite's exact layout.
# ---------------------------------------------------------------------------------------------

def write_summary(directory, metrics, types):
    """Write summary.csv and detailed.csv in the layout write_summary_csv / write_detailed_csv
    produce. `metrics` maps metric -> dict over STATS; `types` maps benchmark_name -> result_type."""
    os.makedirs(directory, exist_ok=True)
    header = list(META_COLUMNS)
    row = ["commit", "0" * 40, "2026-01-01 00:00:00", "2026-01-01T00:00:00"]
    for metric, stats in metrics.items():
        for stat in STATS:
            header.append(f"{metric}_{stat}")
            v = stats[stat]
            row.append(f"{int(v)}" if stat == "samples" else f"{v:.6f}")
    with open(os.path.join(directory, "summary.csv"), "w", newline="") as f:
        f.write(",".join(header) + "\n" + ",".join(row) + "\n")
    with open(os.path.join(directory, "detailed.csv"), "w", newline="") as f:
        f.write("benchmark_name,params,metadata,result_type,samples,outliers_removed,median_us,mad_us,"
                "min_us,max_us,avg_us,stddev_us,ci_lower_us,ci_upper_us,ci_width_percent\n")
        for name, result_type in types.items():
            f.write(f'{name},"p=1",,{result_type},5,0,1,0,1,1,1,0,1,1,0\n')
    return os.path.join(directory, "summary.csv")


def stats_of(avg, samples=10):
    return {"avg_us": avg, "stddev_us": avg * 0.05, "samples": samples,
            "min_us": avg * 0.9, "max_us": avg * 1.2}


def scaled(metrics, metric, factor):
    out = {m: dict(s) for m, s in metrics.items()}
    for stat in TIMING_STATS:
        out[metric][stat] = metrics[metric][stat] * factor
    return out


def self_test():
    types = {
        "evolution_thread_scaling": "TIME",
        "comparative_config1": "TIME",
        "comparative_config1_speedup": "RATIO",
    }
    base = {
        "evolution_thread_scaling_1": stats_of(4000.0),
        "evolution_thread_scaling_4": stats_of(1200.0),
        "comparative_config1_3": stats_of(900.0),
        "comparative_config1_speedup_3": stats_of(5.0),
    }
    slowed_metric = "evolution_thread_scaling_4"
    ratio_metric = "comparative_config1_speedup_3"
    checks = []

    def run(name, head_metrics, expect_code, expect_regressions):
        with tempfile.TemporaryDirectory(prefix="perf_compare_selftest_") as tmp:
            b = write_summary(os.path.join(tmp, "base"), base, types)
            h = write_summary(os.path.join(tmp, "head"), head_metrics, types)
            with open(os.devnull, "w") as sink:
                code, rows = run_compare(b, h, DEFAULT_TOLERANCE, "avg_us", None, out=sink)
        got = [r.metric for r in rows if r.status == "REGRESSION"]
        ok = code == expect_code and got == expect_regressions
        checks.append((name, ok, code, got))

    run("identical files pass", dict(base), 0, [])
    run("20% slowdown fails and names the metric", scaled(base, slowed_metric, 1.20), 1, [slowed_metric])
    run("5% rise passes at the default tolerance", scaled(base, slowed_metric, 1.05), 0, [])
    run("20% fall of a RATIO metric fails", scaled(base, ratio_metric, 0.80), 1, [ratio_metric])
    run("20% rise of a RATIO metric passes", scaled(base, ratio_metric, 1.20), 0, [])
    with_new = dict(base)
    with_new["evolution_thread_scaling_8"] = stats_of(700.0)
    run("a metric only in head is `new`, not a failure", with_new, 0, [])
    without_one = {m: s for m, s in base.items() if m != "comparative_config1_3"}
    run("a metric only in baseline is `missing`, not a failure", without_one, 0, [])

    failed = 0
    for name, ok, code, got in checks:
        print(f"{'PASS' if ok else 'FAIL'}: {name} (exit {code}, regressions {got})")
        failed += 0 if ok else 1
    print(f"self-test: {len(checks) - failed}/{len(checks)} checks passed")
    return 1 if failed else 0


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("baseline", nargs="?", help="baseline summary.csv")
    ap.add_argument("head", nargs="?", help="summary.csv under test")
    ap.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE,
                    help=f"fractional worsening above which a metric regresses (default {DEFAULT_TOLERANCE})")
    ap.add_argument("--stat", choices=TIMING_STATS, default="avg_us", help="the gated statistic (default avg_us)")
    ap.add_argument("--limit", type=int, default=None, help="print only the worst N rows")
    ap.add_argument("--list", metavar="SUMMARY", help="print every parsed metric of one summary.csv and exit")
    ap.add_argument("--self-test", action="store_true", help="run the synthetic self-test and exit")
    args = ap.parse_args(argv)

    try:
        if args.self_test:
            return self_test()
        if args.list:
            list_metrics(args.list)
            return 0
        if not args.baseline or not args.head:
            ap.error("BASELINE_SUMMARY and HEAD_SUMMARY are required (or --list / --self-test)")
        code, _ = run_compare(args.baseline, args.head, args.tolerance, args.stat, args.limit)
        return code
    except SummaryError as e:
        print(f"perf_compare: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
