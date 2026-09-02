"""Aggregate results.tsv over replicates into a readable comparison.

    python summarise.py results/results.tsv

Prints one block per scenario, methods ordered by the headline metric (mean
absolute error on internal node dates, in days), and writes summary.tsv
alongside the input for anything that wants to plot it.
"""

import argparse
import csv
import math
import os
import statistics

# Columns averaged across replicates. Everything else is either an identifier
# or is only meaningful per-run.
AVERAGED = [
    "internal_mae_days",
    "internal_median_ae_days",
    "internal_p90_ae_days",
    "internal_rmse_days",
    "internal_bias_days",
    "internal_within_30d",
    "internal_within_90d",
    "internal_date_correlation",
    "tip_mae_days",
    "undated_tip_mae_days",
    "root_error_days",
    "runtime_seconds",
    "peak_memory_mb",
    "inferred_rate_per_site_per_year",
]


def to_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.fmean(values) if values else None


def fmt(value, places=1):
    if value is None:
        return "-"
    if isinstance(value, float) and math.isnan(value):
        return "-"
    return f"{value:.{places}f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", help="Path to results.tsv")
    parser.add_argument("--sort-by", default="internal_mae_days",
                        help="Metric used to order methods within a scenario")
    args = parser.parse_args()

    with open(args.results) as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    if not rows:
        raise SystemExit("No rows in results file")

    grouped = {}
    for row in rows:
        key = (row["scenario"], row["method"])
        grouped.setdefault(key, []).append(row)

    summary_rows = []
    for (scenario, method), group in grouped.items():
        ok = [r for r in group if r["status"] == "ok"]
        entry = {
            "scenario": scenario,
            "method": method,
            "n_replicates": len(group),
            "n_ok": len(ok),
        }
        for column in AVERAGED:
            entry[column] = mean([to_float(r.get(column)) for r in ok])
        # Spread across replicates tells you whether a gap is real.
        maes = [to_float(r.get("internal_mae_days")) for r in ok]
        maes = [m for m in maes if m is not None]
        entry["internal_mae_days_sd"] = (
            statistics.stdev(maes) if len(maes) > 1 else 0.0 if maes else None)
        summary_rows.append(entry)

    out_path = os.path.join(os.path.dirname(os.path.abspath(args.results)),
                            "summary.tsv")
    fieldnames = (["scenario", "method", "n_replicates", "n_ok",
                   "internal_mae_days_sd"] + AVERAGED)
    with open(out_path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t",
                                extrasaction="ignore")
        writer.writeheader()
        for entry in sorted(summary_rows,
                            key=lambda e: (e["scenario"], e["method"])):
            writer.writerow(entry)

    scenarios = sorted({entry["scenario"] for entry in summary_rows})
    header = (f"{'method':<30}{'MAE':>8}{'±sd':>7}{'median':>8}{'p90':>8}"
              f"{'bias':>8}{'<30d':>7}{'root':>8}{'secs':>8}{'MB':>8}")

    for scenario in scenarios:
        entries = [e for e in summary_rows if e["scenario"] == scenario]
        entries.sort(key=lambda e: (e[args.sort_by] is None,
                                    e[args.sort_by] if e[args.sort_by] is not None else 0))

        print(f"\n{scenario}")
        print("-" * len(header))
        print(header)
        print("-" * len(header))
        for entry in entries:
            failed = entry["n_ok"] < entry["n_replicates"]
            name = entry["method"] + (" *" if failed else "")
            print(f"{name:<30}"
                  f"{fmt(entry['internal_mae_days']):>8}"
                  f"{fmt(entry['internal_mae_days_sd']):>7}"
                  f"{fmt(entry['internal_median_ae_days']):>8}"
                  f"{fmt(entry['internal_p90_ae_days']):>8}"
                  f"{fmt(entry['internal_bias_days']):>8}"
                  f"{fmt(entry['internal_within_30d'], 2):>7}"
                  f"{fmt(entry['root_error_days']):>8}"
                  f"{fmt(entry['runtime_seconds']):>8}"
                  f"{fmt(entry['peak_memory_mb'], 0):>8}")

    if any(e["n_ok"] < e["n_replicates"] for e in summary_rows):
        print("\n* method failed on at least one replicate; "
              "averages cover successful runs only")

    print("\nAll errors are in days, on internal node dates.")
    print("MAE mean absolute error, bias mean signed error (positive is too "
          "recent), <30d fraction of nodes within a month, root error on the "
          "root date.")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
