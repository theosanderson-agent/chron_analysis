"""Verify that node labels mean the same thing in the input and output trees.

The scoring joins inferred dates to truth by node label. That is only sound if
each method returns dates for the tree it was given, with labels still attached
to the same clades. Both treetime and chronumental do preserve them, but that
is a property of the current versions rather than a guarantee, so this checks
it rather than trusting it.

    python check_topology.py --sim-dir results/sims/strict_clean_rep0 \
                             --runs-dir results/runs/strict_clean_rep0

Run it against a benchmark invoked with --keep-working, since it needs the
output trees that a normal run deletes.
"""

import argparse
import os
import sys

import treeswift

import evaluate


def load_tree(path):
    """Read a newick or nexus tree, whichever the tool wrote."""
    if path.endswith((".nexus", ".nex")):
        trees = treeswift.read_tree_nexus(path)
        return trees[list(trees.keys())[0]]
    return treeswift.read_tree_newick(path)


# Where each method leaves its time tree, relative to its run directory.
OUTPUT_TREES = {
    "treetime": "timetree.nexus",
    "treetime-covariation": "timetree.nexus",
    "treetime-relaxed": "timetree.nexus",
    "chronumental": "tree.nwk",
    "chronumental-variance-clock": "tree.nwk",
    "chronumental-horseshoe": "tree.nwk",
    "chronumental-long": "tree.nwk",
    "chronumental-true-clock": "tree.nwk",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-dir", required=True,
                        help="Simulation directory holding divergence_tree.nwk")
    parser.add_argument("--runs-dir", required=True,
                        help="Directory of per-method run directories")
    args = parser.parse_args()

    reference = load_tree(os.path.join(args.sim_dir, "divergence_tree.nwk"))
    reference_signatures = evaluate.clade_signatures(reference)
    print(f"Input tree: {len(reference_signatures)} labelled internal nodes\n")

    failures = 0
    checked = 0

    for method in sorted(os.listdir(args.runs_dir)):
        run_dir = os.path.join(args.runs_dir, method)
        if not os.path.isdir(run_dir):
            continue
        filename = OUTPUT_TREES.get(method)
        if filename is None:
            print(f"{method:<32} no output tree location known, skipped")
            continue
        tree_path = os.path.join(run_dir, filename)
        if not os.path.exists(tree_path):
            print(f"{method:<32} {filename} not found "
                  f"(run the benchmark with --keep-working)")
            continue

        other = load_tree(tree_path)
        shared, mismatched = evaluate.compare_topology(reference, other)
        checked += 1

        missing = len(reference_signatures) - shared
        if mismatched or missing:
            failures += 1
            print(f"{method:<32} FAIL  {shared} shared labels, "
                  f"{mismatched} attached to a different clade, "
                  f"{missing} input labels absent")
        else:
            print(f"{method:<32} ok    all {shared} labels match")

    if not checked:
        print("\nNothing was checked.")
        return 1

    if failures:
        print(f"\n{failures} method(s) failed. Dates cannot be safely joined "
              f"by node name for those, and evaluate.score would silently "
              f"compare the wrong nodes.")
        return 1

    print(f"\nAll {checked} methods preserve node labels. "
          f"Joining dates by name is sound.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
