"""Score inferred dates against the simulated truth.

All errors are reported in days, because that is the unit people actually
reason about when they look at a time tree, and because it keeps scenarios
with different tree heights comparable.

The headline number is the error on *internal* nodes. Tip dates are largely
handed to the methods as input, so scoring them mostly measures how much a
method is willing to move a date it was given. Internal nodes are the part
that has to be inferred.
"""

import math

DAYS_PER_YEAR = 365.25


def read_truth(path):
    """Read simulate.py's truth.tsv into {node: (is_tip, decimal_year)}."""
    truth = {}
    with open(path) as handle:
        header = handle.readline().rstrip("\n").split("\t")
        expected = ["node", "is_tip", "true_decimal_year"]
        if header[:3] != expected:
            raise ValueError(f"Unexpected truth header: {header[:3]}")
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            fields = line.split("\t")
            truth[fields[0]] = (fields[1] == "1", float(fields[2]))
    return truth


def read_dated_tips(path):
    """The tips a method was actually given a date for."""
    dated = set()
    with open(path) as handle:
        handle.readline()
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            name, _, date = line.partition("\t")
            if date.strip():
                dated.add(name)
    return dated


def _summarise(errors_days):
    """Turn a list of signed errors in days into the summary statistics."""
    if not errors_days:
        return {
            "n": 0,
            "mae_days": None,
            "median_ae_days": None,
            "p90_ae_days": None,
            "rmse_days": None,
            "bias_days": None,
            "within_30d": None,
            "within_90d": None,
        }

    absolute = sorted(abs(error) for error in errors_days)
    n = len(absolute)

    def percentile(fraction):
        if n == 1:
            return absolute[0]
        position = fraction * (n - 1)
        lower = int(math.floor(position))
        upper = min(lower + 1, n - 1)
        weight = position - lower
        return absolute[lower] * (1 - weight) + absolute[upper] * weight

    return {
        "n": n,
        "mae_days": sum(absolute) / n,
        "median_ae_days": percentile(0.5),
        "p90_ae_days": percentile(0.9),
        "rmse_days": math.sqrt(sum(e * e for e in errors_days) / n),
        "bias_days": sum(errors_days) / n,
        "within_30d": sum(1 for a in absolute if a <= 30) / n,
        "within_90d": sum(1 for a in absolute if a <= 90) / n,
    }


def _correlation(pairs):
    if len(pairs) < 2:
        return None
    xs = [x for x, _ in pairs]
    ys = [y for _, y in pairs]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x <= 0 or var_y <= 0:
        return None
    return cov / math.sqrt(var_x * var_y)


def score(truth, inferred, dated_tips, root_label):
    """Compare one method's inferred dates against truth.

    Reports internal nodes and tips separately, and singles out tips whose
    dates were withheld, since predicting those is a real inference task
    rather than an echo of the input.
    """
    internal_errors = []
    tip_errors = []
    undated_tip_errors = []
    internal_pairs = []

    matched = 0
    for node, (is_tip, true_year) in truth.items():
        if node not in inferred:
            continue
        matched += 1
        error_days = (inferred[node] - true_year) * DAYS_PER_YEAR
        if is_tip:
            tip_errors.append(error_days)
            if node not in dated_tips:
                undated_tip_errors.append(error_days)
        else:
            internal_errors.append(error_days)
            internal_pairs.append((true_year, inferred[node]))

    n_internal_truth = sum(1 for is_tip, _ in truth.values() if not is_tip)

    result = {
        "n_nodes_matched": matched,
        "n_nodes_truth": len(truth),
        "n_internal_unmatched": n_internal_truth - len(internal_errors),
        "internal_date_correlation": _correlation(internal_pairs),
    }

    for prefix, errors in (("internal", internal_errors),
                           ("tip", tip_errors),
                           ("undated_tip", undated_tip_errors)):
        for key, value in _summarise(errors).items():
            result[f"{prefix}_{key}"] = value

    if root_label in truth and root_label in inferred:
        result["root_error_days"] = (
            (inferred[root_label] - truth[root_label][1]) * DAYS_PER_YEAR)
    else:
        result["root_error_days"] = None

    return result


def clade_signatures(tree):
    """Map each internal node label to the set of tip labels beneath it.

    Used to confirm that a method returned dates for the same tree we handed
    it, rather than a relabelled or rerooted one. Name-based matching is only
    trustworthy if this holds.
    """
    signatures = {}

    for node in tree.traverse_postorder():
        if node.is_leaf():
            node.tip_set = frozenset([node.label])
        else:
            tips = set()
            for child in node.children:
                tips |= child.tip_set
            node.tip_set = frozenset(tips)
            if node.label:
                signatures[node.label.replace("'", "")] = node.tip_set

    return signatures


def compare_topology(reference_tree, other_tree):
    """Check that two trees give the same tip set to each shared node label.

    Returns (n_shared_labels, n_mismatched). A non-zero mismatch means dates
    cannot safely be joined by node name.
    """
    reference = clade_signatures(reference_tree)
    other = clade_signatures(other_tree)
    shared = set(reference) & set(other)
    mismatched = sum(1 for label in shared if reference[label] != other[label])
    return len(shared), mismatched
