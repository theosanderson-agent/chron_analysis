# Brief for agents improving chronumental

Goal: improve chronumental's **predictive accuracy** or its **speed**. Nothing
else counts. A change that is merely tidier is not an improvement.

You are working in your own git worktree of the chronumental repository. Other
agents are working on other hypotheses at the same time, in their own
worktrees. Do not touch anything outside your worktree except to read.

## What chronumental does

It converts a divergence tree (branch lengths in mutations, or in mutations per
site with `--treat_mutation_units_as_normalised_to_genome_size`) into a time
tree, by fitting a model with stochastic variational inference in numpyro. The
code is small: `src/chronumental/__main__.py` holds the CLI and fitting loop,
`models.py` the two models, `helpers.py` a sparse matmul.

## How to measure

**Always measure before and after. Claims without numbers are worthless.**

Set up an environment for your worktree:

```bash
cd <your worktree>
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e .
```

The benchmark harness lives at `/home/theo/chron_analysis/simulation/`. It
simulates time trees where the true date of every internal node is known, which
is what makes accuracy measurable at all. It invokes whatever `chronumental` is
first on `PATH`, so put your venv there:

```bash
cd /home/theo/chron_analysis/simulation
PATH=<your worktree>/.venv/bin:$PATH \
  /home/theo/chron_analysis/simulation/.venv-tools/bin/python run_benchmark.py \
  --out-dir /tmp/<your name>_after \
  --scenarios strict_clean relaxed_clock hard \
  --replicates 2 --n-tips 200 --methods chronumental --jobs 3
```

That is six runs and takes roughly ten minutes. Summarise with:

```bash
/home/theo/chron_analysis/simulation/.venv-tools/bin/python summarise.py /tmp/<your name>_after/results.tsv
```

The headline metric is `internal_mae_days`, the mean absolute error in days on
**internal** node dates. Tip dates are given to the method as input, so tip
error measures obedience, not inference. Lower is better.

A shared baseline for exactly that command, using unmodified chronumental, is
at `/home/theo/chron_analysis/simulation/baseline_dev/`. Compare against it.
If it is missing, generate your own baseline from an unmodified checkout first.

### On timing measurements

Several agents and a large benchmark may be running at once, so wall-clock
numbers are contended and unreliable. If your hypothesis is about speed, also
report a low-contention measurement: run the same command twice on a quiet
machine, or count SVI steps per second, and say which you did.

## Ground rules

- **Correctness first.** A faster or lower-error version that changes the model
  into something else is not an improvement. If you change the model's meaning,
  say so explicitly and loudly.
- **Do not overfit to the simulation.** The harness generates mutations from a
  Poisson model, which is chronumental's own likelihood. Tuning constants until
  this specific benchmark improves is not a real gain. Prefer changes that are
  principled.
- **Keep the CLI compatible.** Existing flags must keep working and keep
  meaning what they meant. New behaviour goes behind a new flag, or becomes the
  default only if it is clearly better across scenarios.
- **Small, reviewable diffs.** One hypothesis per worktree.
- **Report honestly.** If your idea did not work, say so with the numbers. A
  well-measured negative result is genuinely useful and will be reported as
  such. Do not oversell a change that is within noise.

## What to report back

1. What you changed, and why you expected it to help.
2. Before and after numbers from the harness, per scenario.
3. Whether the difference exceeds replicate-to-replicate spread.
4. Any risk or behaviour change a reviewer should know about.
5. The exact `git diff` of your worktree, or the files you changed.
