# Working notes

Last updated 2026-09-02.

Two related threads are in flight. The first got chronumental running again on
current dependencies. The second builds a simulation benchmark to measure how
well it dates trees, compared against treetime. The first had to happen before
the second was possible, because chronumental would not start at all.

---

## 1. chronumental runs on current dependencies again

**Where it lives.** Branch `v2`, commit `943b4e5`. Pushed to
`theosanderson-agent/chronumental`, not to the main repository, because the
account being used has read-only access there. No pull request has been opened.
Getting this onto `theosanderson/chronumental` needs either push access granted
to that account, or a PR from the fork.

**Three code bugs blocked modern versions.**

- `jax.lib.xla_bridge` was removed from jax, so the program died on import
  before doing any work. It now calls `jax.default_backend()`.
- numpyro reordered `TruncatedNormal` from `(low, loc, scale)` to
  `(loc, scale, *, low=)` in its 0.9.0 release. This was confirmed by reading
  the source out of numpyro wheels from 0.6 through 0.19 rather than guessed,
  which dates the breakage to early 2022. The consequence is that
  `--variance_on_clock_rate` has been crashing for about four years.
- `--always_use_final_params` raised `UnboundLocalError`, because `params` was
  only ever assigned in the opposite branch of the `if`.

**One behaviour change to be aware of.** The output tree used to take branch
times from the final optimiser parameters while the output dates came from the
best-loss parameters. Both now come from the same set, which is what the flag
name implies. On a normally converging run these are identical, so this should
not move any published result.

**Packaging was quietly broken too.** A static `version = 0.0.6` in `setup.cfg`
was overriding setuptools_scm, so the version file was never written and the
tool always announced itself as "Chronumental dev" regardless of the release.
Removing it lets the version come from git tags. This is why the publish
workflow now needs `fetch-depth: 0`: without full history there are no tags to
derive a version from.

**Version floors moved.** Python goes from 3.6 to 3.11, since current jax,
numpy and scipy require 3.12 and numpyro and pandas require 3.11. jax is now a
declared dependency rather than something inherited through numpyro. Tested on
3.11, 3.12 and 3.13.

**The yapf pre-commit hook had to be replaced.** The `mirrors-yapf` repository
is archived and frozen at yapf 0.32.0, which imports the `lib2to3` module that
has since been removed from the standard library, so the hook could not run at
all. It now comes from upstream `google/yapf`. Running it reformatted the
source, which is why the diff in `models.py` and `input_mod.py` is larger than
the actual fixes.

---

## 2. Simulation benchmark: treetime vs chronumental

Lives in `simulation/`. See `simulation/README.md` for how to run it.

### Why

`pruning/` measures runtime and memory as tree size grows. `subsets/` measures
agreement with Nextstrain dates on real SARS-CoV-2 data. Neither can score
internal node dates, because no real dataset knows them. Simulating the tree is
the only way to get ground truth for the nodes that actually have to be
inferred.

### How it works

`simulate.py` builds a genealogy with a serially-sampled coalescent, so tips
spread across a sampling window and every node carries a true date. It then
converts elapsed time on each branch into divergence by drawing a Poisson
number of mutations, and writes branch lengths in substitutions per site.

Methods see the divergence tree and the tip dates. They never see `truth.tsv`.

Six scenarios. The first is the case chronumental's default model is exactly
right about, since Poisson mutations on a strict clock is its own likelihood.
The other five each break one assumption, so a change in the ranking can be
attributed to a cause rather than guessed at.

| Scenario | What it breaks |
|---|---|
| `strict_clean` | Nothing |
| `relaxed_clock` | Per-branch rates vary lognormally |
| `noisy_dates` | Date noise, many dates only to month or year |
| `missing_dates` | Thirty percent of tips undated |
| `sparse_sampling` | Short sampling window, rate weakly identifiable |
| `hard` | All of the above together |

Eight methods: three treetime variants (plain, `--covariation`, `--relax`) and
five chronumental configurations (default, learnt clock variance, horseshoe, a
long run, and one handed the true clock rate as a best case).

### Decisions worth remembering

**The headline metric is error on internal nodes, not tips.** Tip dates are
handed to the methods as input, and both reproduce them to within half a day.
Scoring tips would measure obedience rather than inference.

**treetime runs with `--keep-root` throughout.** chronumental never reroots, so
letting treetime pick a better root would stop the two solving the same problem.

**Results are joined to truth by node label.** This was verified, not assumed.
All eight methods preserve the internal labels of the input tree, checked by
comparing the tip set below every labelled node. `check_topology.py` re-runs
that check, and fails loudly if a future version of either tool starts
renaming or rerooting.

### Caveat that limits what the numbers mean

The simulation generates divergence directly rather than evolving sequences and
reconstructing a phylogeny. Every method therefore receives the true topology,
with no reconstruction error and no unresolved polytomies. This isolates the
dating problem, which is what we want to measure, but it makes the absolute
error figures optimistic for everyone. The comparison between methods is the
meaningful part, not the absolute numbers.

Both tools also receive the true genome length, so neither is penalised for
mis-specifying the scale that converts per-site branch lengths to mutations.

---

## Current state (2026-09-02, after the improvement swarm)

### Branch `v2-perf`: seven merged improvements

All measured on simulated trees where every internal node's true date is known,
and all verified independently of the agent that proposed them.

| Commit | Change | Effect |
|---|---|---|
| `ef73b40` | SVI steps in chunks under `lax.scan` | 15-30x faster, accuracy identical |
| `9212644` | Theil-Sen instead of least squares for the starting clock rate | relaxed-clock mean error 176 to 35 days, worst 691 to 51 |
| `53e6f41` | `--tip_date_init`, tip-date-informed initialisation (opt-in) | 30% lower error, better in 14 of 15 replicates |
| `b5e513f` | Early stopping on date stability, on device (default on) | 21-65% fewer steps, saving grows with tree size |
| `77ef5c5` | Convergence check samples nodes rather than using all | gave back ~1 GB at 100k tips (later made moot by `bbb66c8`) |
| `9677df6` | Iterative tree traversal | fixes a hard `RecursionError` crash at 1M tips |
| `73d2448` | Ignore the root's branch length | real-data bug: ebola median 18.5 to 12.6 days vs treetime |
| `bbb66c8` | Pointer-jumping path sums, not a sparse matrix | 8.7x faster and 4.4x smaller at 300k tips |

Two things about the speed work are worth keeping. First, chronumental is now
**faster than treetime**, roughly 4 seconds against 25-31 on a 300-tip tree,
where before this it took 180. Second, the mechanism was not what it looked
like: removing the `np.isnan` sync was worth 20%, removing every host sync from
the Python loop only 26%, and essentially all the gain came from `lax.scan`
eliminating per-step Python dispatch. Do not re-litigate the syncs.

Early stopping being on by default turns `--steps` from a promise into a
ceiling. That is a real if narrow compatibility break, mitigated by
`--disable_early_stopping`. The answer is verifiably unchanged, which is why it
was allowed to default on where `--tip_date_init`, which does change answers,
was not.

### chronumental now handles a million tips

This is the headline. At the start of this work chronumental could not date a
million-tip tree at all, which is the size it exists for. Two separate defects
stood in the way, both found here.

`9677df6`: `helpers.preorder_traversal` recursed once per tree level, so a deep
tree exhausted Python's call stack. A million-tip coalescent tree raised
`RecursionError` before fitting began.

`bbb66c8`: root-to-tip sums used a sparse matrix with one entry per
(node, ancestor) pair, so its memory grew with the total path length summed
over the tree. At 300k tips that was 116 million entries and 2.6 GB of a
3.2 GB run, the largest single allocation. Pointer jumping computes the same
sums in memory proportional to node count.

Whole-run, 2000-step ceiling:

| Tips | before | after |
|---|---|---|
| 100,000 | 77 s, 1.95 GB | 16 s, 1.10 GB |
| 300,000 | 415 s, 6.83 GB | 48 s, 1.54 GB |
| 1,000,000 | never finished, killed at 33 GB | **219 s, 3.84 GB** |

At a million tips all 999,999 internal nodes are dated, with mean absolute
error 29.2 days, median 11.4, 90th percentile 70.6, 77.5% within a month, and
the root 70 days out. The median is *better* than the 300-tip benchmark's 14.6
days, because a denser tree constrains each node with more nearby tips.

Pointer jumping also suits a GPU better, since it needs a gather and an add
where the sparse form needed a scatter-add. On an RTX 2080 Ti at 100k tips the
path sum uses 2.7 MB of device memory against 183 MB, and takes 0.10 ms against
1.53 ms. GPU memory being scarcer than host memory, the old structure was
closer to fatal there than on CPU.

### Real data: recapitulating treetime on ebola

Everything above is simulation, where the generator is ours. The ebola example
dataset is the check against that. Using treetime's published dates as the
reference, over 228 internal nodes:

| | median \|diff\| | mean signed | max | root date |
|---|---|---|---|---|
| LSD2 | 8.1 d | -3.4 d | 83 d | 2013-10-15 |
| chronumental before `73d2448` | 18.5 d | +20.3 d | 183 d | 2014-04-02 |
| chronumental after | 12.6 d | +10.4 d | 75 d | 2013-12-14 |
| treetime | - | - | - | 2013-10-01 |

The fix was that a root has no parent, so a branch length on it spans no time,
but treetime's divergence trees carry one and chronumental was treating it as
elapsed time. **The simulation benchmark could never have caught this**, because
its generated trees give the root no branch length. It only appears on trees
produced by other tools, which is chronumental's normal input.

Further chasing treetime's numbers on this dataset would be misguided: treetime
and LSD2 disagree with each other by 24% on the clock rate, and chronumental now
sits inside the range they span. There is no ground truth here to tune toward.

### Benchmark result: where each tool wins

Full run, six scenarios, five replicates, eight methods, 240 runs. Mean
absolute error on internal node dates, in days.

| Scenario | best treetime | chronumental | Winner |
|---|---|---|---|
| `hard` | 103.0 | 38.6 | chronumental, 2.7x |
| `noisy_dates` | 67.4 | 17.6 | chronumental, 3.8x |
| `missing_dates` | 11.5 | 12.7 | treetime, marginal |
| `strict_clean` | 10.7 | 16.0 | treetime |
| `sparse_sampling` | 7.8 | 10.2 | treetime |
| `relaxed_clock` | 17.8 | 44.1 | treetime, 2.5x |

The split is interpretable. chronumental dominates when **date metadata is
bad**, which is the SARS-CoV-2 situation it was built for. treetime wins when
the **clock assumption is violated** or temporal signal is thin.

### The central open problem: the objective does not prefer the right answer

Seeding the fit from the true branch times roughly halves the error, but the
ELBO is **not** better at that far more accurate solution. Tested across 15
replicates, the ELBO preferred the truth-seeded solution in only 1 of 15.

Because the guide is a Delta distribution, maximising the ELBO here is
essentially MAP estimation, so this points at the model rather than the
optimiser. Many very different time trees explain the data about equally well
and nothing in the model prefers the right one. This explains a lot of
otherwise puzzling results: why learning-rate schedules changed nothing, why
10000 steps scores the same as 2000, and why better initialisation helps. It
helps by starting inside the right basin, not by optimising harder.

Note one apparent counter-example that is not one. Forcing the true *clock
rate* does give a better loss, 6723 against 6854. So the objective does
discriminate on the global rate. What it fails to discriminate is the
**allocation of time along a path**, which is what internal node dates are.

### Negative results, all well measured

These were real attempts that did not work. Recorded so nobody repeats them.

- **A relaxed clock does not help.** Branch `worktree-agent-a6f8c5be922897f38`.
  A proper uncorrelated lognormal relaxed clock scores within noise of the
  default on every scenario. The mechanism is instructive: once a branch's rate
  can float, its mutation count is explained by the rate, so the Poisson term
  stops constraining how time is allocated along the path, which is exactly the
  signal internal dating depends on. The catastrophic relaxed-clock failures
  turned out to be a rate *initialisation* problem, fixed by Theil-Sen.
- **A coalescent prior on node times does not help either.** Branch
  `worktree-agent-ac581138368a9c9ab`. This was the obvious fix for the
  identifiability problem, and it is implemented correctly and scales, 2-16%
  overhead. At its natural strength it does not move where SVI converges.
  Weighting it 100x does flip the ELBO's preference toward the truth, which
  proves the prior carries correctly-signed information, but accuracy then gets
  worse because one global Ne cannot tell "correct" from "generically
  coalescent-shaped". Note the simulator generates trees from exactly this
  process, so this was the most favourable possible test, and it still failed.
- **Learning-rate schedules do nothing.** Identical to three significant
  figures on rate and to the day on root error.

### Claims that were checked and turned out to be wrong

Recorded because each was believed for a while.

- **`mutation_rate_sigma` is not collapsing.** It was reported as decaying
  without bound under `--variance_on_clock_rate`. Probed directly: it falls
  from 24 to about 0.71 by step 3000 and is then flat through 6000. That is
  convergence to a genuine posterior width of about 3% of the rate. The ELBO's
  entropy term penalises a shrinking scale, so it cannot run to zero.
- **There is no systematic early bias in the root date.** That came from a
  single replicate. Root error changes sign across replicates and tracks clock
  rate error instead.
- **The nan check was not the cause of the slowness.** See above.

### Real bugs still unfixed

- **`tau_param` in `HorseShoeLike` is dead code.** `fixed_tau` is hardcoded
  `True` in `__main__.py`, so the guide always substitutes `initial_tau` and
  `tau_param` never enters the computation graph or receives a gradient. It may
  have been pinned as an undocumented workaround for a variance-component
  collapse, since a learnt point estimate for a HalfCauchy scale, whose mode is
  at zero, would be genuinely unstable.
- **The horseshoe model is the worst chronumental variant on every scenario**
  in the benchmark, which given it exists to tolerate bad date metadata and
  loses on `noisy_dates` deserves investigation.

### Next steps

- Decide whether `--tip_date_init` should become the default. The simulation
  evidence supports it, but the simulator uses chronumental's own likelihood,
  so a check against real data should come first.
- Fix or remove the dead `tau_param`, and work out why the horseshoe model
  underperforms.
- The identifiability problem is the ceiling on accuracy. A coalescent prior at
  natural strength does not fix it. Worth considering: a non-Delta guide, so
  the ELBO sees posterior width rather than doing MAP; or a prior on
  branch-time *ratios* rather than absolute tree shape.
- Get `v2` and `v2-perf` onto the main repository. The account in use has
  read-only access, so this needs push rights or a pull request from the fork.

## 3. Initialisation of the variational parameters is an identifiability problem, not a convergence problem

From the worktree investigating whether chronumental's starting values for
`time_length_mu` and `root_date_mu` matter (crude default: mutations divided
by clock rate, floored at a few days; ignores tip dates entirely).

### The finding that matters more than the flag it produced

Seeding the fit from the **true** simulated branch times (diagnostic only,
never shipped, no real run has this available) cut `internal_mae_days`
several-fold at the benchmark's normal step count, on every scenario tested,
while the **final ELBO loss was often not better** — sometimes worse. Example,
hard scenario, 200-tip replicate: truth-seeded run converged to a *higher*
loss (6914.93 vs 6879.67) but roughly halved the error (10.1 vs 21.4 days).
On strict_clean and relaxed_clock replicates the effect was larger still
(13.4 to 1.6 days, and 21.9 to 1.5 days respectively).

This says the variational objective does not reliably prefer the date
solution that is actually correct. The model can represent answers several
times better than what SVI finds from a cold start, and 2000 steps of Adam is
enough time to find them if you start nearby, but not enough to be pulled
there from the default start point, because the objective does not pull that
way. That reframes what "improving accuracy" is fighting: not slow
convergence, not model expressiveness, but a Poisson-clock likelihood that
under-constrains node dates enough for a factually wrong branch-time
configuration to score as good as, or better than, a correct one.

**A second, cleaner example turned up after rebasing onto `v2-perf`** (Theil-Sen
clock-rate initialisation, see below), which rules out "the starting rate was
also wrong" as the explanation. `relaxed_clock` replicate 4, 300 tips:
Theil-Sen gave a starting rate of 24.68/genome/year against a true 24.0, i.e.
already correct to within 3%. The default (crude time) initialisation still
converged to a mutation rate of **11.0**, less than half the true value, at a
loss of 10442.9. `--tip_date_init`, started from the same correct rate but
better branch times, held near the true rate (~27) throughout and finished at
a *higher* loss of 18662.5, with `internal_mae_days` of 18.2 against the
default run's 117.9. The objectively wrong answer has the lower loss. This
is not an edge case produced by hunting for one; it was the single largest
delta in a five-replicate sweep, not a cherry-pick.

### Practical consequence

`--tip_date_init` (opt-in flag, not default) seeds branch times and the root
date from a tip-date-consistent estimate (tree + tip dates + starting clock
rate; no ground truth) rather than the crude formula, with shrinkage toward
the global average for weakly-supported clades and a floor tied to each
branch's own mutation count so real mutations never get a near-zero initial
time. It cannot fix non-identifiability in general, but it starts the
optimiser closer to correct answers, which the diagnostic above shows matters
independently of how good the starting rate is.

### Re-measured on `v2-perf` (scan loop + Theil-Sen rate), 5 replicates/scenario, 300 tips

Rebased cleanly onto `v2-perf` (ef73b40, 9212644), no conflicts. Ran
chronumental twice per sim (default vs `--tip_date_init`, `--steps 2000`)
against `simulation/results/sims/{strict_clean,relaxed_clock,hard}_rep{0..4}`.
`internal_mae_days`, lower is better:

| scenario | rep0 | rep1 | rep2 | rep3 | rep4 | mean (baseline → flag) |
|---|---|---|---|---|---|---|
| strict_clean | 12.49→12.38 | 18.24→16.07 | 23.24→17.77 | 10.01→10.57 | 16.23→14.36 | 16.04 → 14.23 |
| relaxed_clock | 15.69→14.58 | 19.60→17.94 | 17.49→13.71 | 49.94→28.66 | 117.92→18.18 | 44.13 → 18.61 |
| hard | 30.07→27.20 | 63.74→62.20 | 33.27→30.81 | 26.66→23.75 | 39.22→34.64 | 38.59 → 35.72 |

`--tip_date_init` won 14 of 15 replicates; the one loss (strict_clean rep3,
+0.55 days) is noise-level. Mean of scenario means: 32.9 → 22.9 days, roughly
a 30% reduction, with no scenario made worse on average.

**Does it help most where the Theil-Sen rate fix helps least?** Not in a clean
monotonic way by how far off the starting rate was — relaxed_clock rep4 had
the *most accurate* starting rate of the five (3% off) and still the *largest*
win from `--tip_date_init` (117.9 → 18.2). rep3 had the least accurate
starting rate (32% off, and in the opposite direction from rep0/1/2) and the
second-largest win (49.9 → 28.7). The pattern that does hold: the win scales
with how badly the *default* run already fails, largely independent of
whether that failure traces back to the rate or not. The two fixes look
orthogonal, addressing different failure modes that happen to compound.

**Kept opt-in, default unchanged**, per instruction — this is a promising
result on n=5, not yet the "clearly better on every scenario at scale" bar
for flipping a default, and strict_clean is a wash rather than a win.

