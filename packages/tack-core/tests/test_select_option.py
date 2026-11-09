# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The root conftest's ``--select``, alone and under ``--shard``: what CI's
test selection relies on (``tools/testmap.py``, ci.yml's ``changes`` job)."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
_FILES = ["packages/tack-core/tests/test_constant.py", "packages/tack-core/tests/test_matrix.py",
          "packages/tack-core/tests/test_cpu_pool.py"]


def _pytest(tmp_path, selection, *args, files=_FILES):
    chosen = tmp_path / "selection.txt"
    chosen.write_text("".join(f"{line}\n" for line in selection))
    run = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                          "--collect-only", f"--select={chosen}", *args, *files],
                         cwd=ROOT, capture_output=True, text=True, timeout=300)
    files = {line.split("::")[0] for line in run.stdout.splitlines() if "::" in line}
    return run.returncode, files


def test_the_shares_divide_the_selection(tmp_path):
    chosen = _FILES[:2]
    shares = [_pytest(tmp_path, chosen, "--shard", f"{k}/2") for k in (1, 2)]
    assert all(code == 0 for code, _ in shares)
    assert shares[0][1] | shares[1][1] == set(chosen)
    assert not shares[0][1] & shares[1][1]


def test_all_keeps_every_file_and_nothing_selected_passes(tmp_path):
    assert _pytest(tmp_path, ["all"]) == (0, set(_FILES))
    assert _pytest(tmp_path, []) == (0, set())


def test_a_selection_outside_the_repository_with_no_paths(tmp_path):
    """As CI runs it: no test paths, the selection in the runner's temporary
    directory. Given as ``--select FILE``, pytest took FILE for a test path
    and its directory's parent for the rootdir, and never read this conftest."""
    assert _pytest(tmp_path, _FILES[:1], files=()) == (0, set(_FILES[:1]))
