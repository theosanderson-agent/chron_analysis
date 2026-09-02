# Simulation benchmark: treetime vs chronumental

Dating methods can only be scored properly against a tree whose internal node
dates are actually known. Real datasets never give us that, so this directory
simulates time trees with known dates, derives the divergence tree they imply,
and asks each method to recover what it started from.

This complements the two evaluations already in the repository. `pruning/`
measures runtime and memory as tree size grows, and `subsets/` measures
agreement with Nextstrain dates on real SARS-CoV-2 data. Neither has access to
ground truth for internal nodes; this does.

## Running it

Both tools install into one environment on current Python:

```bash
python -m venv .venv
.venv/bin/pip install phylo-treetime
.venv/bin/pip install /path/to/chronumental
```

Then check the harness runs end to end, which takes a couple of minutes:

```bash
.venv/bin/python run_benchmark.py --out-dir results_quick --quick
```

And run it properly:

```bash
.venv/bin/python run_benchmark.py --out-dir results
.venv/bin/python summarise.py results/results.tsv
```

`--quick` uses one replicate, 120 tips and two scenarios. The full run is six
scenarios by three replicates by eight methods.

## What is simulated

`simulate.py` builds a genealogy with a serially-sampled coalescent, so tips
are spread across a sampling window and every node carries a true date. It then
walks the branches and converts elapsed time into divergence: each branch draws
a Poisson number of mutations from its rate and duration, and branch lengths
are written in substitutions per site.

That last step deliberately matches chronumental's own Poisson likelihood. On
the `strict_clean` scenario chronumental's default model is therefore exactly
correct, which is the fairest possible ground for it. The other scenarios each
break one assumption, so a change in the ranking can be attributed to a cause
rather than guessed at.

| Scenario | What it breaks |
|---|---|
| `strict_clean` | Nothing. Strict clock, exact daily dates. |
| `relaxed_clock` | Per-branch rates vary lognormally. |
| `noisy_dates` | Two weeks of date noise, many dates only to month or year. |
| `missing_dates` | Thirty percent of tips have no date. |
| `sparse_sampling` | A short sampling window, so the rate is weakly identifiable. |
| `hard` | All of the above at once. |

Three files come out of each simulation. `divergence_tree.nwk` and `dates.tsv`
are what the methods see. `truth.tsv` holds the true date of every node and is
never passed to a method.

## What is compared

Eight methods, defined in `methods.py`. TreeTime runs with `--keep-root`
throughout, because chronumental never reroots, and letting one method choose
a better root would not be measuring the same thing.

- `treetime`, `treetime-covariation`, `treetime-relaxed`
- `chronumental` (the default strict-clock model), `chronumental-variance-clock`,
  `chronumental-horseshoe`, `chronumental-long`
- `chronumental-true-clock`, which is handed the true simulated rate and told
  to hold it fixed. It is a best case rather than a fair competitor, and is
  there to show how much of the remaining error comes from misjudging the
  clock rate as opposed to everything else.

Adding a method means adding one entry to `METHODS`. Nothing else changes.

## How the scoring works

The headline metric is mean absolute error on **internal** node dates, in days.
Tip dates are mostly handed to the methods as input, so scoring those largely
measures how far a method will move a date it was already given. Internal nodes
are the part that has to be inferred.

Also reported: median and 90th percentile error, which say whether a method is
uniformly mediocre or usually good with a bad tail; signed bias, which catches
a method that is systematically too recent or too old; the error on the root
date specifically, which is the number most often quoted from a time tree; and
the fraction of nodes landing within a month. Runtime and peak memory come from
`/usr/bin/time`, the same way `pruning/performance.py` measures them.

Methods are joined to truth by node label. Both tools preserve the internal
labels in the input tree, which was verified rather than assumed, and
`check_topology.py` re-checks it by comparing the tip set below each labelled
node in the output tree against the input. If a future version of either tool
starts renaming or rerooting, that check fails loudly instead of silently
comparing the wrong nodes.

A method that crashes or times out is recorded with a status rather than
dropped, since a method failing on a scenario is itself a result.

## Caveats

The simulation generates divergence directly from a Poisson model rather than
evolving sequences along the tree and reconstructing a phylogeny. So the
divergence tree handed to each method has the true topology, with no
reconstruction error and no unresolved polytomies. This isolates the dating
problem, which is what we want to measure here, but it means the absolute error
figures are optimistic for every method. The comparison between methods is the
meaningful part, not the absolute numbers.

Both tools also receive the true genome length, so neither is penalised for
mis-specifying the scale that converts per-site branch lengths into mutations.
