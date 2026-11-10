# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""tools/testmap.py: merging records, reading a diff, selecting test files."""

import testmap as tm

_CORE = "packages/tack-core/tests/"
_VIS = "packages/tack-vis/tests/"
_MAP = {"tests": {
    _VIS + "test_clip.py": ["packages/tack-vis/src/tack/dataset/algorithms/clip.py",
                            "packages/tack-core/src/tack/lang/kernel.py"],
    _VIS + "test_probe.py": ["packages/tack-vis/src/tack/dataset/algorithms/probe.py",
                             "packages/tack-core/src/tack/lang/kernel.py"],
    _CORE + "test_scan.py": ["packages/tack-core/src/tack/algorithms/scan.py",
                             "packages/tack-core/src/tack/lang/kernel.py"],
}}


def _select(*changed):
    return tm.select(list(changed), _MAP)


def test_records_merge_by_test_file():
    merged = tm.merge([{"a.py": ["x", "y"]}, {"a.py": ["y", "z"], "b.py": ["x"]}])
    assert merged == {"a.py": ["x", "y", "z"], "b.py": ["x"]}


def test_a_diff_is_read_with_renames_as_a_deletion_and_an_addition():
    lines = ["M\tsrc/a.py\n", "A\tsrc/new.py\n", "D\tsrc/old.py\n", "R087\tsrc/b.py\tsrc/c.py\n"]
    assert tm.changes(lines) == [("src/a.py", "M"), ("src/new.py", "A"), ("src/old.py", "D"),
                                 ("src/b.py", "D"), ("src/c.py", "A")]


def test_a_source_file_selects_the_test_files_that_ran_it():
    assert _select(("packages/tack-vis/src/tack/dataset/algorithms/clip.py", "M")) == (
        [_VIS + "test_clip.py"], None)
    tests, why = _select(("packages/tack-core/src/tack/lang/kernel.py", "M"))
    assert len(tests) == 3 and why is None


def test_a_test_file_selects_itself_unless_deleted():
    assert _select((_CORE + "test_new.py", "A")) == ([_CORE + "test_new.py"], None)
    assert _select((_CORE + "test_scan.py", "D")) == ([], None)


def test_build_and_ci_files_select_everything():
    for path in ("pyproject.toml", "packages/tack-core/pyproject.toml", "uv.lock", "conftest.py",
                 "packages/tack-vis/tests/conftest.py", ".github/workflows/ci.yml",
                 "test-durations.json"):
        tests, why = _select((path, "M"))
        assert tests == tm.ALL and path in why, path


def test_an_unmapped_file_selects_everything_unless_the_change_adds_it():
    tests, why = _select(("packages/tack-vis/src/tack/dataset/algorithms/unused.py", "M"))
    assert tests == tm.ALL and "unused.py" in why
    tests, why = _select(("packages/tack-vis/tests/data/mesh.vtu", "M"))
    assert tests == tm.ALL, "data a test reads is not in the map"
    assert _select(("packages/tack-vis/src/tack/dataset/algorithms/new.py", "A"),
                   ("packages/tack-vis/src/tack/dataset/algorithms/probe.py", "M")) == (
        [_VIS + "test_probe.py"], None)


def test_docs_examples_and_tools_select_nothing():
    assert _select(("docs/index.md", "M"), ("tools/changelog.py", "M"),
                   ("packages/tack-core/examples/01_hello_tack.py", "M")) == ([], None)
