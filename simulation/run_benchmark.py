"""Run every method over every scenario and write one tidy table of results.

Usage:

    python run_benchmark.py --out-dir results/            # the full run
    python run_benchmark.py --out-dir results/ --quick    # a fast smoke test
    python run_benchmark.py --out-dir results/ --jobs 4   # four methods at once

Output is results.tsv, one row per (scenario, replicate, method). Rows for
runs that failed are kept, with a status column saying so, because a method
falling over on a scenario is itself a finding.

On --jobs: accuracy is unaffected by running methods concurrently, but the
runtime and memory columns are only comparable at --jobs 1. The table records
which was used. `pruning/performance.py` is the place to look for careful
timing anyway.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor

import evaluate
import methods as methods_module

# Each scenario is a set of overrides passed to simulate.py. The first is the
# case chronumental's default model is exactly right about; the rest break one
# assumption each, so that a difference in the results can be attributed.
SCENARIOS = {
    "strict_clean": {
        "description": "Strict clock, exact daily tip dates. chronumental's "
                       "default model is correctly specified here.",
        "args": {},
    },
    "relaxed_clock": {
        "description": "Per-branch rates vary lognormally, breaking the "
                       "strict-clock assumption.",
        "args": {"relaxed-sigma": 0.5},
    },
    "noisy_dates": {
        "description": "Tip dates carry two weeks of noise and many are only "
                       "reported to the month or year.",
        "args": {
            "date-noise-days": 14,
            "month-precision-fraction": 0.3,
            "year-precision-fraction": 0.1,
        },
    },
    "missing_dates": {
        "description": "Thirty percent of tips have no date at all.",
        "args": {"missing-date-fraction": 0.3},
    },
    "sparse_sampling": {
        "description": "A short sampling window, so the clock rate is only "
                       "weakly identifiable.",
        "args": {"sampling-window": 0.5},
    },
    "hard": {
        "description": "Relaxed clock, noisy and imprecise dates, and missing "
                       "dates together. The realistic case.",
        "args": {
            "relaxed-sigma": 0.5,
            "date-noise-days": 14,
            "month-precision-fraction": 0.3,
            "year-precision-fraction": 0.1,
            "missing-date-fraction": 0.2,
        },
    },
}

RESULT_COLUMNS = [
    "scenario", "replicate", "seed", "method", "status",
    "n_tips", "n_internal_nodes", "genome_length",
    "true_clock_rate_per_site_per_year", "inferred_rate_per_site_per_year",
    "runtime_seconds", "peak_memory_mb", "parallel_jobs",
    "internal_mae_days", "internal_median_ae_days", "internal_p90_ae_days",
    "internal_rmse_days", "internal_bias_days",
    "internal_within_30d", "internal_within_90d",
    "internal_date_correlation",
    "tip_mae_days", "tip_median_ae_days",
    "undated_tip_n", "undated_tip_mae_days",
    "root_error_days",
    "n_nodes_matched", "n_internal_unmatched",
    "notes",
]


def simulate_scenario(scenario_name, scenario, sim_dir, seed, n_tips,
                      genome_length):
    """Invoke simulate.py for one scenario and replicate."""
    here = os.path.dirname(os.path.abspath(__file__))
    command = [
        sys.executable, os.path.join(here, "simulate.py"),
        "--out-dir", sim_dir,
        "--seed", str(seed),
        "--n-tips", str(n_tips),
        "--genome-length", str(genome_length),
    ]
    for key, value in scenario["args"].items():
        command += [f"--{key}", str(value)]

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Simulation failed for {scenario_name} seed {seed}:\n{result.stderr}")
    with open(os.path.join(sim_dir, "simulation.json")) as handle:
        return json.load(handle)


def root_label_from_truth(truth_path):
    """The first row simulate.py writes is the root, since it writes preorder."""
    with open(truth_path) as handle:
        handle.readline()
        return handle.readline().split("\t")[0]


def tidy_run_dir(out_dir):
    """Drop bulky per-method outputs, keeping the logs someone would read."""
    for name in os.listdir(out_dir):
        if name in ("stdout.txt", "stderr.txt"):
            continue
        path = os.path.join(out_dir, name)
        if os.path.isfile(path):
            os.remove(path)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark treetime against chronumental on simulated trees.")
    parser.add_argument("--out-dir", default="results",
                        help="Where to write results.tsv and the working files")
    parser.add_argument("--scenarios", nargs="*", default=None,
                        help="Subset of scenarios to run (default: all)")
    parser.add_argument("--methods", nargs="*", default=None,
                        help="Subset of methods to run (default: all)")
    parser.add_argument("--replicates", type=int, default=3,
                        help="Independent simulated datasets per scenario")
    parser.add_argument("--n-tips", type=int, default=300)
    parser.add_argument("--genome-length", type=int, default=30000)
    parser.add_argument("--timeout", type=int, default=3600,
                        help="Per-method timeout in seconds")
    parser.add_argument("--jobs", type=int, default=1,
                        help="Methods to run concurrently. Timing columns are "
                             "only comparable at 1.")
    parser.add_argument("--keep-working", action="store_true",
                        help="Keep each method's raw output directory")
    parser.add_argument("--quick", action="store_true",
                        help="One replicate, 120 tips, two scenarios. For "
                             "checking the harness runs end to end.")
    args = parser.parse_args()

    if args.quick:
        args.replicates = 1
        args.n_tips = 120
        if args.scenarios is None:
            args.scenarios = ["strict_clean", "hard"]

    scenario_names = args.scenarios or list(SCENARIOS)
    unknown = set(scenario_names) - set(SCENARIOS)
    if unknown:
        parser.error(f"Unknown scenarios: {sorted(unknown)}")

    if args.methods:
        unknown = set(args.methods) - set(methods_module.METHODS_BY_NAME)
        if unknown:
            parser.error(f"Unknown methods: {sorted(unknown)}")
        selected = [methods_module.METHODS_BY_NAME[name] for name in args.methods]
    else:
        selected = methods_module.METHODS

    os.makedirs(args.out_dir, exist_ok=True)
    results_path = os.path.join(args.out_dir, "results.tsv")

    # Simulate every dataset up front. This is fast, and doing it first means
    # a failure in the simulator surfaces before any long method run starts.
    print("Simulating datasets ...")
    datasets = []
    for scenario_name in scenario_names:
        scenario = SCENARIOS[scenario_name]
        # Seed from the scenario's own name, not its position in whatever
        # subset was requested. Deriving it from an index meant that running
        # `--scenarios strict_clean hard` produced different trees than a full
        # run did for those same two scenarios, so absolute numbers were not
        # comparable between runs that selected different subsets. Comparisons
        # within a run were always fine, since every method saw the same data.
        scenario_seed = 1000 * (1 + zlib.crc32(scenario_name.encode()) % 1000)
        for replicate in range(args.replicates):
            seed = scenario_seed + replicate
            sim_dir = os.path.join(args.out_dir, "sims",
                                   f"{scenario_name}_rep{replicate}")
            sim_meta = simulate_scenario(scenario_name, scenario, sim_dir,
                                         seed, args.n_tips, args.genome_length)
            truth_path = os.path.join(sim_dir, "truth.tsv")
            datasets.append({
                "scenario": scenario_name,
                "replicate": replicate,
                "seed": seed,
                "sim_dir": sim_dir,
                "sim_meta": sim_meta,
                "truth": evaluate.read_truth(truth_path),
                "dated_tips": evaluate.read_dated_tips(
                    os.path.join(sim_dir, "dates.tsv")),
                "root_label": root_label_from_truth(truth_path),
            })
    print(f"  {len(datasets)} datasets, {args.n_tips} tips each")

    tasks = [(dataset, method) for dataset in datasets for method in selected]
    total = len(tasks)
    print(f"Running {total} method runs with --jobs {args.jobs}\n")

    write_lock = threading.Lock()
    counter = {"done": 0}

    with open(results_path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS,
                                delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        handle.flush()

        def execute(task):
            dataset, method = task
            out_dir = os.path.join(
                args.out_dir, "runs",
                f"{dataset['scenario']}_rep{dataset['replicate']}", method.name)

            run = methods_module.run_method(
                method, dataset["sim_dir"], out_dir, args.genome_length,
                dataset["sim_meta"], timeout_seconds=args.timeout)

            sim_meta = dataset["sim_meta"]
            row = {
                "scenario": dataset["scenario"],
                "replicate": dataset["replicate"],
                "seed": dataset["seed"],
                "method": method.name,
                "status": run["status"],
                "n_tips": sim_meta["n_tips"],
                "n_internal_nodes": sim_meta["n_internal_nodes"],
                "genome_length": args.genome_length,
                "true_clock_rate_per_site_per_year":
                    sim_meta["clock_rate_per_site_per_year"],
                "runtime_seconds": run.get("runtime_seconds"),
                "peak_memory_mb": run.get("peak_memory_mb"),
                "parallel_jobs": args.jobs,
                "notes": run.get("stderr_tail", ""),
            }

            summary = run["status"].upper()
            if run["status"] == "ok":
                inferred = run["inferred_dates"]
                # A parser may date the root without being able to name it.
                sentinel = methods_module.ROOT_SENTINEL
                if sentinel in inferred:
                    value = inferred.pop(sentinel)
                    inferred.setdefault(dataset["root_label"], value)
                scores = evaluate.score(dataset["truth"], inferred,
                                        dataset["dated_tips"],
                                        dataset["root_label"])
                row.update(scores)
                row["inferred_rate_per_site_per_year"] = run.get(
                    "inferred_rate_per_site_per_year")
                if scores["n_internal_unmatched"]:
                    row["notes"] = (
                        f"{scores['n_internal_unmatched']} internal nodes had "
                        f"no inferred date; check node labelling")
                mae = scores["internal_mae_days"]
                summary = (f"internal MAE {mae:.1f} d, "
                           f"{run['runtime_seconds']:.0f} s")
                if not args.keep_working:
                    tidy_run_dir(out_dir)

            with write_lock:
                counter["done"] += 1
                writer.writerow(row)
                handle.flush()
                print(f"[{counter['done']}/{total}] "
                      f"{dataset['scenario']} rep{dataset['replicate']} "
                      f"{method.name}: {summary}", flush=True)

        if args.jobs > 1:
            with ThreadPoolExecutor(max_workers=args.jobs) as pool:
                list(pool.map(execute, tasks))
        else:
            for task in tasks:
                execute(task)

    print(f"\nWrote {results_path}")
    print(f"Summarise it with: python summarise.py {results_path}")


if __name__ == "__main__":
    main()
