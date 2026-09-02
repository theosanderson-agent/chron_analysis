"""The dating methods under comparison, and how to run and read each one.

Every method takes the same three inputs (a divergence tree with per-site
branch lengths, a tip date file, and the genome length) and produces the same
output: a mapping from node label to inferred date in decimal years. That
uniformity is what lets the scoring code stay method-agnostic.

Adding a method means adding one entry to METHODS. Nothing else needs to change.
"""

import datetime
import os
import re
import shlex
import subprocess
import sys
import time

DAYS_PER_YEAR = 365.25

# Key a parser may use for a node it dates but cannot name. LSD2 emits the root
# this way. run_benchmark maps it onto the real root label before scoring.
ROOT_SENTINEL = "__unlabelled_root__"


def date_to_decimal_year(date):
    """Inverse of simulate.decimal_year_to_date."""
    year = date.year
    start = datetime.datetime(year=year, month=1, day=1)
    end = datetime.datetime(year=year + 1, month=1, day=1)
    if isinstance(date, datetime.datetime):
        moment = date
    else:
        moment = datetime.datetime(date.year, date.month, date.day)
    return year + (moment - start).total_seconds() / (end - start).total_seconds()


def parse_treetime_dates(out_dir):
    """Read treetime's dates.tsv.

    The file has a '#node date numeric date' header and gives the inferred
    date for every node, internal ones included.
    """
    path = os.path.join(out_dir, "dates.tsv")
    result = {}
    with open(path) as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) < 3:
                continue
            name, _date, numeric = fields[0], fields[1], fields[2]
            try:
                result[name] = float(numeric)
            except ValueError:
                continue
    return result


def parse_chronumental_dates(out_dir):
    """Read the TSV chronumental writes to --dates_out."""
    path = os.path.join(out_dir, "dates.tsv")
    result = {}
    with open(path) as handle:
        header = handle.readline()
        if "predicted_date" not in header:
            raise ValueError(f"Unexpected chronumental header: {header!r}")
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            name, _, value = line.partition("\t")
            value = value.strip()
            if not value:
                continue
            # chronumental writes a full timestamp, e.g. 2017-11-17 02:03:53.322
            try:
                moment = datetime.datetime.fromisoformat(value)
            except ValueError:
                moment = datetime.datetime.strptime(value[:10], "%Y-%m-%d")
            result[name] = date_to_decimal_year(moment)
    return result


def parse_treetime_rate(out_dir):
    """Pull the fitted clock rate out of treetime's molecular_clock.txt."""
    path = os.path.join(out_dir, "molecular_clock.txt")
    if not os.path.exists(path):
        return None
    text = open(path).read()
    match = re.search(r"Root-Tip-Regression:\s*--rate:\s*([0-9.eE+-]+)", text)
    if match:
        return float(match.group(1))
    match = re.search(r"rate:\s*([0-9.eE+-]+)", text)
    return float(match.group(1)) if match else None


def parse_chronumental_rate(out_dir):
    """Pull the final mutation rate out of chronumental's captured stdout.

    chronumental logs a tab-separated progress line every ten steps; the last
    one carries the fitted rate in mutations per genome per year.
    """
    path = os.path.join(out_dir, "stdout.txt")
    if not os.path.exists(path):
        return None
    rate = None
    for line in open(path):
        match = re.search(r"mutation_rate:([0-9.eE+-]+)", line)
        if match:
            rate = float(match.group(1))
    return rate


def parse_lsd2_dates(out_dir):
    """Read node dates out of LSD2's annotated nexus tree.

    LSD2 writes the tree with each node's date as a nexus comment attached to
    its label, like `node_00038[&date="2021-03-04"]`. With -D 2 those dates are
    to the day, which the default two-decimal year format is not.
    """
    path = os.path.join(out_dir, "lsd2.date.nexus")
    text = open(path).read()
    result = {}

    def parse(value):
        value = value.strip()
        if "-" in value:
            return date_to_decimal_year(
                datetime.datetime.strptime(value[:10], "%Y-%m-%d"))
        return float(value)

    # LSD2 dates the root but writes it without a label, as a bare
    # `)[&date="..."]` at the end of the tree. Report it under a sentinel the
    # caller maps onto the real root label, so root error stays comparable.
    root = re.search(r'\)\[&date="([^"]+)"\]\s*;', text)
    if root:
        try:
            result[ROOT_SENTINEL] = parse(root.group(1))
        except ValueError:
            pass

    for name, date in re.findall(r'([A-Za-z0-9_.\-]+)\[&date="([^"]+)"\]', text):
        date = date.strip()
        try:
            if "-" in date:
                moment = datetime.datetime.strptime(date[:10], "%Y-%m-%d")
                result[name] = date_to_decimal_year(moment)
            else:
                result[name] = float(date)
        except ValueError:
            continue
    return result


def parse_lsd2_rate(out_dir):
    """LSD2 prints the fitted rate on its results line."""
    path = os.path.join(out_dir, "stdout.txt")
    if not os.path.exists(path):
        return None
    match = re.search(r"rate ([0-9.eE+-]+)", open(path).read())
    return float(match.group(1)) if match else None


def _lsd2_command(extra_args):
    """LSD2 needs its own date format, so convert first, then run.

    -l -1 disables LSD2's default collapsing of near-zero-length internal
    branches. That default is reasonable for LSD2 on its own, but it merges
    internal nodes, and we can only score a node whose date every method
    reports. Disabling it keeps all methods dating the identical tree. It also
    happens to make LSD2 more accurate here, so this is not handicapping it.
    """
    here = os.path.dirname(os.path.abspath(__file__))

    def build(sim_dir, out_dir, genome_length, sim_meta):
        dates_in = os.path.join(sim_dir, "dates.tsv")
        dates_out = os.path.join(out_dir, "lsd2_dates.txt")
        convert = (f"{shlex.quote(sys.executable)} "
                   f"{shlex.quote(os.path.join(here, 'lsd2_dates.py'))} "
                   f"{shlex.quote(dates_in)} {shlex.quote(dates_out)}")
        run = (f"lsd2 -i {shlex.quote(os.path.join(sim_dir, 'divergence_tree.nwk'))}"
               f" -d {shlex.quote(dates_out)}"
               f" -s {genome_length}"
               f" -o {shlex.quote(os.path.join(out_dir, 'lsd2'))}"
               f" -l -1 -D 2 {extra_args}")
        return f"{convert} && {run}"

    return build


class Method:
    def __init__(self, name, family, build_command, parse_dates,
                 parse_rate=None, rate_units="per_genome_per_year",
                 description=""):
        self.name = name
        self.family = family
        self.build_command = build_command
        self.parse_dates = parse_dates
        self.parse_rate = parse_rate
        self.rate_units = rate_units
        self.description = description


def _treetime_command(extra_args):
    def build(sim_dir, out_dir, genome_length, sim_meta):
        return (
            f"treetime --tree {shlex.quote(os.path.join(sim_dir, 'divergence_tree.nwk'))}"
            f" --dates {shlex.quote(os.path.join(sim_dir, 'dates.tsv'))}"
            f" --sequence-length {genome_length}"
            f" --keep-root"
            f" --outdir {shlex.quote(out_dir)}"
            f" {extra_args}"
        )
    return build


def _chronumental_command(extra_args, use_true_clock=False):
    def build(sim_dir, out_dir, genome_length, sim_meta):
        clock = ""
        if use_true_clock:
            # chronumental multiplies --clock by the genome size, so it wants
            # the per-site rate here, matching the simulation parameter.
            clock = f" --clock {sim_meta['clock_rate_per_site_per_year']}"
        return (
            f"chronumental --tree {shlex.quote(os.path.join(sim_dir, 'divergence_tree.nwk'))}"
            f" --dates {shlex.quote(os.path.join(sim_dir, 'dates.tsv'))}"
            f" --treat_mutation_units_as_normalised_to_genome_size {genome_length}"
            f" --dates_out {shlex.quote(os.path.join(out_dir, 'dates.tsv'))}"
            f" --tree_out {shlex.quote(os.path.join(out_dir, 'tree.nwk'))}"
            f"{clock} {extra_args}"
        )
    return build


DEFAULT_STEPS = 2000

METHODS = [
    Method(
        name="treetime",
        family="treetime",
        build_command=_treetime_command(""),
        parse_dates=parse_treetime_dates,
        parse_rate=parse_treetime_rate,
        rate_units="per_site_per_year",
        description="TreeTime with the input rooting kept, so it solves the "
                    "same problem chronumental does.",
    ),
    Method(
        name="treetime-covariation",
        family="treetime",
        build_command=_treetime_command("--covariation"),
        parse_dates=parse_treetime_dates,
        parse_rate=parse_treetime_rate,
        rate_units="per_site_per_year",
        description="TreeTime accounting for shared ancestry when it fits the "
                    "clock rate.",
    ),
    Method(
        name="treetime-relaxed",
        family="treetime",
        build_command=_treetime_command("--relax 1.0 0.5"),
        parse_dates=parse_treetime_dates,
        parse_rate=parse_treetime_rate,
        rate_units="per_site_per_year",
        description="TreeTime with an autocorrelated relaxed clock.",
    ),
    Method(
        name="chronumental",
        family="chronumental",
        build_command=_chronumental_command(f"--steps {DEFAULT_STEPS}"),
        parse_dates=parse_chronumental_dates,
        parse_rate=parse_chronumental_rate,
        description="chronumental's default strict-clock model.",
    ),
    Method(
        name="chronumental-variance-clock",
        family="chronumental",
        build_command=_chronumental_command(
            f"--steps {DEFAULT_STEPS} --variance_on_clock_rate"),
        parse_dates=parse_chronumental_dates,
        parse_rate=parse_chronumental_rate,
        description="Clock rate drawn from a distribution with a learnt variance.",
    ),
    Method(
        name="chronumental-horseshoe",
        family="chronumental",
        build_command=_chronumental_command(
            f"--steps {DEFAULT_STEPS} --model HorseShoeLike"),
        parse_dates=parse_chronumental_dates,
        parse_rate=parse_chronumental_rate,
        description="Horseshoe-like per-tip date variance, meant to tolerate "
                    "bad date metadata.",
    ),
    Method(
        name="chronumental-long",
        family="chronumental",
        build_command=_chronumental_command("--steps 10000"),
        parse_dates=parse_chronumental_dates,
        parse_rate=parse_chronumental_rate,
        description="The default model run five times longer, to separate "
                    "model error from unconverged optimisation.",
    ),
    Method(
        name="chronumental-true-clock",
        family="chronumental",
        build_command=_chronumental_command(
            f"--steps {DEFAULT_STEPS} --enforce_exact_clock",
            use_true_clock=True),
        parse_dates=parse_chronumental_dates,
        parse_rate=parse_chronumental_rate,
        description="Given the true simulated clock rate and told to hold it "
                    "fixed. A best case, not a fair competitor.",
    ),
    Method(
        name="lsd2",
        family="lsd2",
        build_command=_lsd2_command(""),
        parse_dates=parse_lsd2_dates,
        parse_rate=parse_lsd2_rate,
        rate_units="per_site_per_year",
        description="LSD2, least-squares dating, keeping the input root.",
    ),
    Method(
        name="lsd2-variance",
        family="lsd2",
        build_command=_lsd2_command("-v 2"),
        parse_dates=parse_lsd2_dates,
        parse_rate=parse_lsd2_rate,
        rate_units="per_site_per_year",
        description="LSD2 re-estimating branch variances from a first pass, "
                    "which its docs suggest for non-strict clocks.",
    ),
    Method(
        name="lsd2-outliers",
        family="lsd2",
        build_command=_lsd2_command("-e 3"),
        parse_dates=parse_lsd2_dates,
        parse_rate=parse_lsd2_rate,
        rate_units="per_site_per_year",
        description="LSD2 excluding tips whose residuals look like date "
                    "errors. Its counterpart to chronumental's horseshoe.",
    ),
]

METHODS_BY_NAME = {method.name: method for method in METHODS}


def run_method(method, sim_dir, out_dir, genome_length, sim_meta,
               timeout_seconds=3600):
    """Run one method, capturing wall time, peak memory, and its output.

    Returns a dict describing the run. A method that fails is recorded rather
    than raised, so one crash does not abandon the whole benchmark.
    """
    os.makedirs(out_dir, exist_ok=True)
    command = method.build_command(sim_dir, out_dir, genome_length, sim_meta)

    # /usr/bin/time reports peak resident set size in kilobytes on the last
    # line of stderr. The existing pruning/performance.py measures memory the
    # same way.
    wrapped = f"/usr/bin/time -f '%M' {command}"

    start = time.time()
    try:
        process = subprocess.run(wrapped, shell=True, capture_output=True,
                                 text=True, timeout=timeout_seconds)
        timed_out = False
    except subprocess.TimeoutExpired:
        return {
            "method": method.name,
            "status": "timeout",
            "runtime_seconds": timeout_seconds,
            "peak_memory_mb": None,
            "command": command,
        }
    runtime = time.time() - start

    with open(os.path.join(out_dir, "stdout.txt"), "wt") as handle:
        handle.write(process.stdout)
    with open(os.path.join(out_dir, "stderr.txt"), "wt") as handle:
        handle.write(process.stderr)

    peak_memory_mb = None
    stderr_lines = [line for line in process.stderr.strip().split("\n") if line.strip()]
    if stderr_lines:
        try:
            peak_memory_mb = int(stderr_lines[-1].strip()) / 1024.0
        except ValueError:
            peak_memory_mb = None

    if process.returncode != 0:
        return {
            "method": method.name,
            "status": "failed",
            "returncode": process.returncode,
            "runtime_seconds": runtime,
            "peak_memory_mb": peak_memory_mb,
            "command": command,
            "stderr_tail": "\n".join(stderr_lines[-5:]),
        }

    try:
        dates = method.parse_dates(out_dir)
    except (OSError, ValueError) as error:
        return {
            "method": method.name,
            "status": "unparseable",
            "runtime_seconds": runtime,
            "peak_memory_mb": peak_memory_mb,
            "command": command,
            "stderr_tail": str(error),
        }

    rate = None
    if method.parse_rate is not None:
        try:
            rate = method.parse_rate(out_dir)
        except (OSError, ValueError):
            rate = None

    # Report every rate in the same units so the column is comparable.
    rate_per_site = None
    if rate is not None:
        if method.rate_units == "per_site_per_year":
            rate_per_site = rate
        else:
            rate_per_site = rate / genome_length

    return {
        "method": method.name,
        "status": "ok",
        "runtime_seconds": runtime,
        "peak_memory_mb": peak_memory_mb,
        "command": command,
        "inferred_dates": dates,
        "inferred_rate_per_site_per_year": rate_per_site,
    }
