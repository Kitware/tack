# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""tools/ci_failures.py: reading job logs, flagging failures, the report."""

import ci_failures as cf

_LOG = """\
2026-11-08T18:53:39.3387999Z =========================== short test summary info ============================
2026-11-08T18:53:39.3387999Z FAILED packages/tack-core/tests/test_cpu_threading.py::test_timing_a_contended_kernel_settles_serial - AssertionError: fanned out 8 times in 12
2026-11-08T18:53:39.3388999Z ERROR packages/tack-vis/tests/test_probe.py::test_probe[cpu-hex] - RuntimeError
2026-11-08T18:53:39.3389999Z FAILED packages/tack-core/tests/test_cpu_threading.py::test_timing_a_contended_kernel_settles_serial - again
2026-11-08T18:53:39.3389837Z 1 failed, 1520 passed, 104 skipped in 722.34s (0:12:02)
"""


def test_failing_tests_are_read_from_the_short_summary_once_each():
    assert cf.failing_tests(_LOG) == [
        "packages/tack-core/tests/test_cpu_threading.py::test_timing_a_contended_kernel_settles_serial",
        "packages/tack-vis/tests/test_probe.py::test_probe[cpu-hex]",
    ]
    assert cf.failing_tests("1520 passed in 700s\n") == []


def test_flaky_tests_are_read_from_the_timing_jobs_warning():
    log = ("2026-11-09T10:00:00.0Z ##[warning]failed, then passed when run again: "
           "a.py::test_one b.py::test_two\n")
    assert cf.flaky_tests(log) == ["a.py::test_one", "b.py::test_two"]
    assert cf.flaky_tests(_LOG) == []


def _failure(test, branch="topic", event="pull_request", sha="s1", job="tests (py3.13, 3/4)",
             run=1, attempt=1, flake=False):
    return cf.Failure(test, run, attempt, job, event, branch, sha, "2026-11-08T18:00:00Z", flake)


def test_flags():
    failures = [
        _failure("t::main", branch="main", event="push"),
        _failure("t::rerun", sha="s2", job="mpi"),
        _failure("t::spread", branch="a"),
        _failure("t::spread", branch="b", run=2),
        _failure("t::flake", job="timing", flake=True),
        _failure("t::plain", sha="s3"),
    ]
    jobs = [("s2", "mpi", "failure"), ("s2", "mpi", "success"),
            ("s3", "tests (py3.13, 3/4)", "failure"), ("s3", "tests (py3.12, 3/4)", "success")]
    rows = {row.test: row.flags for row in cf.classify(failures, jobs)}
    assert rows == {"t::main": {"main"}, "t::rerun": {"re-run"}, "t::spread": {"branches"},
                    "t::flake": {"flake"}, "t::plain": set()}


def test_rows_with_more_flags_come_first():
    failures = [_failure("t::plain", sha="s3"), _failure("t::main", branch="main", event="push"),
                _failure("t::main", branch="topic", run=2)]
    assert [row.test for row in cf.classify(failures, [])] == ["t::main", "t::plain"]


def test_report():
    rows = cf.classify([_failure("t::x", branch="main", event="push", run=42, attempt=2)], [])
    text = cf.report(rows, "Kitware/tack", 7, 10, "2026-11-09")
    assert "1 failures of 1 tests" in text
    assert "| `t::x` | 1 | main | `main` | [2026-11-08](https://github.com/Kitware/tack/actions/runs/42/attempts/2) |" in text
    assert "No test failed." in cf.report([], "Kitware/tack", 7, 10, "2026-11-09")


def test_unflagged_rows_fold_away_below_the_flagged():
    rows = cf.classify([_failure("t::main", branch="main", event="push"),
                        _failure("t::plain", sha="s3")], [])
    text = cf.report(rows, "Kitware/tack", 7, 10, "2026-11-09")
    assert text.index("t::main") < text.index("<details>") < text.index("t::plain")
    assert "1 tests failed on one branch alone" in text
    only_plain = cf.report(cf.classify([_failure("t::plain")], []), "Kitware/tack", 7, 10, "2026-11-09")
    assert "No failure is flagged." in only_plain
