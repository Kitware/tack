# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Which test files a change can reach: the map of what each test file runs,
and the selection a pull request's changed files make from it.

The map is recorded on ``main`` (``tools/testmap_recorder/``: a pytest plugin, with
``tools/testmap_recorder`` on ``PYTHONPATH``)::

    TESTMAP_OUT=out PYTHONPATH=tools/testmap_recorder uv run pytest -p testmap_plugin ...
    python3 tools/testmap.py merge out -o test-map.json --commit SHA

and CI publishes it as the ``test-map`` artifact of every run of ``ci.yml``
(the wiki's CI test management plan, step 5). A selection reads the changed
files as ``git diff --name-status`` prints them::

    git diff --name-status BASE HEAD | python3 tools/testmap.py select test-map.json

and prints the selected test files, one a line, or ``all`` when the change
needs the whole suite. The rules, first match per file:

- a file the path filter (``tools/ci_changes.py``) puts in no test job
  (docs, examples, tools) selects nothing;
- a build or CI file (``pyproject.toml``, ``uv.lock``, a ``conftest.py``,
  ``.github/``, ``test-durations.json``) selects everything;
- a test file selects itself, unless the change deletes it;
- a file the map has selects the test files that ran it;
- a file the change adds selects nothing by itself: the changed files that
  use it select theirs;
- any other file (one no test ran, data a test reads) selects everything.
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ci_changes

ALL = "all"
_BUILD = ("pyproject.toml", "uv.lock", "test-durations.json")


def merge(records):
    """One map from the processes' records: ``{test file: [files]}``."""
    used = {}
    for record in records:
        for test, files in record.items():
            used.setdefault(test, set()).update(files)
    return {test: sorted(files) for test, files in sorted(used.items())}


def is_test_file(path):
    name = os.path.basename(path)
    return "/tests/" in f"/{path}" and name.startswith("test_") and name.endswith(".py")


def _is_build(path):
    return (os.path.basename(path) in _BUILD or os.path.basename(path) == "conftest.py"
            or path.startswith(".github/"))


def changes(lines):
    """``(path, status)`` from ``git diff --name-status`` lines; a rename is
    its old path deleted and its new one added."""
    out = []
    for line in lines:
        fields = line.rstrip("\n").split("\t")
        if len(fields) < 2:
            continue
        status = fields[0][:1]
        if status == "R" and len(fields) == 3:
            out += [(fields[1], "D"), (fields[2], "A")]
        elif status == "C" and len(fields) == 3:
            out.append((fields[2], "A"))
        else:
            out.append((fields[1], status))
    return out


def select(changed, test_map):
    """The test files ``changed`` (``(path, status)`` pairs) reaches, sorted,
    and why not when it needs them all: ``(tests, None)`` or ``(ALL, reason)``."""
    users = {}
    for test, files in test_map.get("tests", {}).items():
        for f in files:
            users.setdefault(f, set()).add(test)
    selected = set()
    for path, status in changed:
        if "code" not in ci_changes.groups_of(path):
            continue
        if _is_build(path):
            return ALL, f"a build or CI file: {path}"
        if is_test_file(path):
            if status != "D":
                selected.add(path)
        elif path in users:
            selected |= users[path]
        elif status == "A":
            continue
        else:
            return ALL, f"no test file in the map runs {path}"
    return sorted(selected), None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    m = commands.add_parser("merge", help="merge the processes' records into a map")
    m.add_argument("records", help="a directory of records (TESTMAP_OUT), read recursively")
    m.add_argument("-o", "--output", required=True)
    m.add_argument("--commit", default="")
    m.add_argument("--least", type=int, default=0,
                   help="fail with fewer test files than this (a broken recorder)")
    s = commands.add_parser("select", help="the test files a change reaches (stdin: git diff --name-status)")
    s.add_argument("map")
    args = parser.parse_args(argv)

    if args.command == "merge":
        records = []
        for path in sorted(glob.glob(os.path.join(args.records, "**", "*.json"), recursive=True)):
            with open(path) as f:
                records.append(json.load(f))
        tests = merge(records)
        with open(args.output, "w") as f:
            json.dump({"commit": args.commit, "tests": tests}, f, indent=0)
        files = {f for used in tests.values() for f in used}
        print(f"{len(records)} records: {len(tests)} test files, {len(files)} files they run",
              file=sys.stderr)
        if len(tests) < args.least:
            raise SystemExit(f"only {len(tests)} test files recorded, fewer than {args.least}")
        return
    with open(args.map) as f:
        test_map = json.load(f)
    tests, why = select(changes(sys.stdin), test_map)
    if why:
        print(f"the whole suite: {why}", file=sys.stderr)
        print(ALL)
    else:
        print(f"{len(tests)} test files", file=sys.stderr)
        for test in tests:
            print(test)


if __name__ == "__main__":
    main()
