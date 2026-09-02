"""Convert the harness date file into the format LSD2 expects.

LSD2 wants a count on the first line, then one "name date" pair per line with
the date as a decimal year. The harness writes a TSV with a header and ISO
dates, which are also the format treetime and chronumental read, so the
conversion has to happen somewhere. Doing it here keeps the method definition
in methods.py to a single shell command.

Partial dates are handled the way the simulator degrades them: a month-only
date becomes the first of that month, a year-only date the first of January.
That is slightly different from chronumental, which centres partial dates in
the interval, but LSD2 has its own bounded-constraint syntax we are not using,
and matching chronumental's centring here would give LSD2 information the
plain format cannot express.
"""

import datetime
import sys


def to_decimal_year(text):
    parts = text.strip().split("-")
    year = int(parts[0])
    month = int(parts[1]) if len(parts) > 1 else 1
    day = int(parts[2]) if len(parts) > 2 else 1
    start = datetime.datetime(year, 1, 1)
    end = datetime.datetime(year + 1, 1, 1)
    moment = datetime.datetime(year, month, day)
    return year + (moment - start).total_seconds() / (end - start).total_seconds()


def main():
    source, destination = sys.argv[1], sys.argv[2]
    rows = []
    with open(source) as handle:
        handle.readline()  # header
        for line in handle:
            name, _, date = line.strip().partition("\t")
            if not date.strip():
                continue
            try:
                rows.append(f"{name} {to_decimal_year(date):.6f}")
            except (ValueError, IndexError):
                # A date LSD2 could not use is better dropped than guessed at;
                # the other methods skip unparseable dates too.
                continue

    with open(destination, "wt") as handle:
        handle.write(f"{len(rows)}\n")
        handle.write("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
