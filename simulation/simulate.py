"""Simulate a time tree with known node dates, then derive a divergence tree.

The point of the simulation is that we know the true date of *every* node,
including internal ones, which is what makes it possible to score dating
methods properly. Real datasets only ever give us tip dates.

The generative process is:

1.  Draw tip sampling times across a sampling window.
2.  Build a genealogy connecting them with a serially-sampled (heterochronous)
    coalescent. This gives every node a true date in decimal years.
3.  Walk the branches and turn elapsed time into genetic divergence. Each
    branch gets a rate (constant under a strict clock, lognormally distributed
    per branch under a relaxed clock) and a Poisson number of mutations.
4.  Emit the divergence tree with per-site branch lengths, a tip date file for
    the dating methods to consume, and a truth file they never see.

Step 3 deliberately matches chronumental's own Poisson likelihood under
`--clock strict`, so the relaxed-clock scenarios exist to make sure we are not
only ever testing the case where chronumental's model is exactly right.
"""

import argparse
import datetime
import json
import math
import random

# A year in days. Dating output is compared in days, so the two conversions
# (decimal year -> date, and date difference -> days) need to agree.
DAYS_PER_YEAR = 365.25


class Node:
    __slots__ = ("label", "parent", "children", "time", "branch_length",
                 "n_mutations", "rate")

    def __init__(self, label=None, time=0.0):
        self.label = label
        self.parent = None
        self.children = []
        self.time = time  # true date, decimal years
        self.branch_length = 0.0  # divergence, substitutions per site
        self.n_mutations = 0
        self.rate = None

    def add_child(self, child):
        child.parent = self
        self.children.append(child)

    def is_leaf(self):
        return not self.children


def preorder(node):
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        stack.extend(reversed(current.children))


def decimal_year_to_date(decimal_year):
    """Convert a decimal year to a datetime, matching chronumental's convention."""
    year = int(math.floor(decimal_year))
    fraction = decimal_year - year
    start = datetime.datetime(year=year, month=1, day=1)
    end = datetime.datetime(year=year + 1, month=1, day=1)
    return start + fraction * (end - start)


def simulate_sampling_times(n_tips, window_years, end_year, rng):
    """Tip sampling times, spread over the sampling window.

    A spread of tip dates is what makes the clock rate identifiable at all, so
    the window is the single most important knob in the simulation.
    """
    return [
        end_year - rng.uniform(0.0, window_years) for _ in range(n_tips)
    ]


def simulate_coalescent(sampling_times, population_size, rng):
    """Serially-sampled coalescent, built backwards in time from the most recent tip.

    `population_size` is a coalescent Ne expressed in years, so that the
    expected time for two lineages to coalesce is Ne years.
    """
    if len(sampling_times) < 2:
        raise ValueError("Need at least two tips to build a tree")

    # Most recent first, so we can pop them off as we walk back in time.
    pending = sorted(
        ((t, Node(label=f"tip_{i:05d}", time=t))
         for i, t in enumerate(sampling_times)),
        key=lambda pair: pair[0],
        reverse=True,
    )

    current_time = pending[0][0]
    active = []
    next_internal = 0

    # Seed with every tip sampled at the most recent time.
    while pending and pending[0][0] >= current_time:
        active.append(pending.pop(0)[1])

    while len(active) > 1 or pending:
        k = len(active)

        if k >= 2:
            rate = k * (k - 1) / (2.0 * population_size)
            wait = rng.expovariate(rate)
            proposed_time = current_time - wait
        else:
            # A single lineage cannot coalesce; jump straight to the next sample.
            proposed_time = -math.inf

        next_sample_time = pending[0][0] if pending else -math.inf

        if next_sample_time > proposed_time:
            # Another tip is sampled before the proposed coalescence.
            current_time = next_sample_time
            while pending and pending[0][0] >= current_time:
                active.append(pending.pop(0)[1])
            continue

        # A coalescence happens. Pick two distinct lineages by index and
        # remove them by swapping with the end of the list, which keeps this
        # O(1) per event instead of O(k) for rng.sample plus list.remove.
        current_time = proposed_time
        i = rng.randrange(k)
        j = rng.randrange(k - 1)
        if j >= i:
            j += 1
        if i < j:
            i, j = j, i
        left = active[i]
        active[i] = active[-1]
        active.pop()
        right = active[j]
        active[j] = active[-1]
        active.pop()
        parent = Node(label=f"node_{next_internal:05d}", time=current_time)
        next_internal += 1
        parent.add_child(left)
        parent.add_child(right)
        active.append(parent)

    return active[0]


def assign_divergence(root, clock_rate, genome_length, rng,
                      relaxed_sigma=0.0):
    """Turn elapsed time on each branch into substitutions per site.

    Under a strict clock every branch shares `clock_rate`. Under a relaxed
    clock each branch draws its own rate from a lognormal with median
    `clock_rate`, which breaks the strict-clock assumption that chronumental's
    default model makes.
    """
    for node in preorder(root):
        if node.parent is None:
            node.branch_length = 0.0
            node.n_mutations = 0
            node.rate = clock_rate
            continue

        elapsed = node.time - node.parent.time
        if elapsed < 0:
            raise AssertionError("Child is older than its parent")

        if relaxed_sigma > 0:
            # Median-preserving lognormal, so the overall pace of the clock is
            # unchanged and only its branch-to-branch variability grows.
            rate = clock_rate * math.exp(rng.gauss(0.0, relaxed_sigma))
        else:
            rate = clock_rate

        expected = rate * elapsed * genome_length
        n_mutations = poisson(expected, rng)

        node.rate = rate
        node.n_mutations = n_mutations
        node.branch_length = n_mutations / genome_length


def poisson(mean, rng):
    """Draw a Poisson variate. Falls back to a normal approximation when large."""
    if mean <= 0:
        return 0
    if mean < 30:
        # Knuth's algorithm.
        limit = math.exp(-mean)
        k = 0
        p = 1.0
        while True:
            p *= rng.random()
            if p <= limit:
                return k
            k += 1
    value = int(round(rng.gauss(mean, math.sqrt(mean))))
    return max(value, 0)


class _Separator:
    """Stack marker that emits a comma between sibling subtrees."""
    label = ","
    branch_length = 0.0
    children = ()

    def is_leaf(self):
        return False


_COMMA = _Separator()


def write_newick(root, path, include_internal_labels=True):
    """Write the divergence tree.

    Internal node labels are written so that outputs can be matched back to
    truth by name where a tool preserves them. Matching also has a name-free
    fallback in evaluate.py, because not every tool does.
    """

    # Written iteratively rather than recursively: a coalescent tree with a
    # few hundred thousand tips is deep enough to exhaust the Python stack,
    # and large trees are the case this whole exercise is about.
    pieces = []
    stack = [(root, False)]
    while stack:
        node, expanded = stack.pop()
        if node is _COMMA:
            pieces.append(",")
            continue
        if node.is_leaf():
            pieces.append(f"{node.label}:{node.branch_length:.10f}")
            continue
        if not expanded:
            stack.append((node, True))
            # Reversed, and with separators interleaved, so the pieces come
            # off the stack in the order they must appear in the string.
            children = node.children
            for index, child in enumerate(reversed(children)):
                if index:
                    stack.append((_COMMA, False))
                stack.append((child, False))
            pieces.append("(")
        else:
            label = node.label if include_internal_labels else ""
            pieces.append(f"){label}:{node.branch_length:.10f}")

    with open(path, "wt") as handle:
        handle.write("".join(pieces) + ";\n")


def degrade_date(true_time, precision, rng, noise_days):
    """Return the date string a method gets to see for a tip.

    Real metadata is not a clean ISO date for every sample: some records only
    give a month or a year, and some are simply wrong. `precision` chooses the
    reported granularity and `noise_days` adds error on top.
    """
    observed = true_time + rng.gauss(0.0, noise_days / DAYS_PER_YEAR)
    date = decimal_year_to_date(observed)

    if precision == "day":
        return date.strftime("%Y-%m-%d")
    if precision == "month":
        return date.strftime("%Y-%m")
    if precision == "year":
        return date.strftime("%Y")
    raise ValueError(f"Unknown precision {precision}")


def choose_precision(rng, month_fraction, year_fraction):
    draw = rng.random()
    if draw < year_fraction:
        return "year"
    if draw < year_fraction + month_fraction:
        return "month"
    return "day"


def main():
    parser = argparse.ArgumentParser(
        description="Simulate a time tree and the divergence tree implied by it.")
    parser.add_argument("--out-dir", required=True,
                        help="Directory to write the simulated dataset into")
    parser.add_argument("--n-tips", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--genome-length", type=int, default=30000)
    parser.add_argument("--clock-rate", type=float, default=8e-4,
                        help="Substitutions per site per year")
    parser.add_argument("--population-size", type=float, default=2.0,
                        help="Coalescent Ne, in years")
    parser.add_argument("--sampling-window", type=float, default=2.0,
                        help="Tip dates are spread over this many years")
    parser.add_argument("--end-year", type=float, default=2021.7,
                        help="Decimal year of the most recent tip")
    parser.add_argument("--relaxed-sigma", type=float, default=0.0,
                        help="Lognormal sd of per-branch rate. 0 is a strict clock")
    parser.add_argument("--date-noise-days", type=float, default=0.0,
                        help="Gaussian noise added to reported tip dates")
    parser.add_argument("--month-precision-fraction", type=float, default=0.0,
                        help="Fraction of tips reported only to the month")
    parser.add_argument("--year-precision-fraction", type=float, default=0.0,
                        help="Fraction of tips reported only to the year")
    parser.add_argument("--missing-date-fraction", type=float, default=0.0,
                        help="Fraction of tips with no date at all")
    args = parser.parse_args()

    import os
    os.makedirs(args.out_dir, exist_ok=True)

    rng = random.Random(args.seed)

    sampling_times = simulate_sampling_times(
        args.n_tips, args.sampling_window, args.end_year, rng)
    root = simulate_coalescent(sampling_times, args.population_size, rng)
    assign_divergence(root, args.clock_rate, args.genome_length, rng,
                      relaxed_sigma=args.relaxed_sigma)

    tree_path = os.path.join(args.out_dir, "divergence_tree.nwk")
    write_newick(root, tree_path)

    # What the methods get to see: tip labels and (possibly degraded) dates.
    dates_path = os.path.join(args.out_dir, "dates.tsv")
    n_missing = 0
    with open(dates_path, "wt") as handle:
        handle.write("strain\tdate\n")
        for node in preorder(root):
            if not node.is_leaf():
                continue
            if rng.random() < args.missing_date_fraction:
                n_missing += 1
                continue
            precision = choose_precision(rng, args.month_precision_fraction,
                                         args.year_precision_fraction)
            date = degrade_date(node.time, precision, rng,
                                args.date_noise_days)
            handle.write(f"{node.label}\t{date}\n")

    # The truth file, which no method is allowed to read.
    truth_path = os.path.join(args.out_dir, "truth.tsv")
    n_internal = 0
    n_tips = 0
    with open(truth_path, "wt") as handle:
        handle.write("node\tis_tip\ttrue_decimal_year\ttrue_date\t"
                     "n_mutations\tbranch_rate\n")
        for node in preorder(root):
            is_tip = node.is_leaf()
            n_tips += is_tip
            n_internal += not is_tip
            date = decimal_year_to_date(node.time)
            rate = "" if node.rate is None else f"{node.rate:.10g}"
            handle.write(
                f"{node.label}\t{int(is_tip)}\t{node.time:.10f}\t"
                f"{date.strftime('%Y-%m-%d')}\t{node.n_mutations}\t{rate}\n")

    total_mutations = sum(n.n_mutations for n in preorder(root))
    root_to_tip = []
    for node in preorder(root):
        if node.is_leaf():
            divergence = 0.0
            walk = node
            while walk.parent is not None:
                divergence += walk.branch_length
                walk = walk.parent
            root_to_tip.append(divergence)

    meta = {
        "n_tips": n_tips,
        "n_internal_nodes": n_internal,
        "n_tips_without_date": n_missing,
        "seed": args.seed,
        "genome_length": args.genome_length,
        "clock_rate_per_site_per_year": args.clock_rate,
        "clock_rate_per_genome_per_year": args.clock_rate * args.genome_length,
        "population_size_years": args.population_size,
        "sampling_window_years": args.sampling_window,
        "relaxed_sigma": args.relaxed_sigma,
        "date_noise_days": args.date_noise_days,
        "month_precision_fraction": args.month_precision_fraction,
        "year_precision_fraction": args.year_precision_fraction,
        "missing_date_fraction": args.missing_date_fraction,
        "root_true_decimal_year": root.time,
        "root_true_date": decimal_year_to_date(root.time).strftime("%Y-%m-%d"),
        "tree_height_years": max(n.time for n in preorder(root)) - root.time,
        "total_mutations": total_mutations,
        "mean_root_to_tip_divergence": sum(root_to_tip) / len(root_to_tip),
    }
    with open(os.path.join(args.out_dir, "simulation.json"), "wt") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)

    print(f"Wrote {n_tips} tips and {n_internal} internal nodes to {args.out_dir}")
    print(f"  tree height {meta['tree_height_years']:.2f} years, "
          f"{total_mutations} mutations, "
          f"root {meta['root_true_date']}")


if __name__ == "__main__":
    main()
