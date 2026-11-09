# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A test's subprocesses record into its test file's map too: with
``tools/testmap_recorder`` on ``PYTHONPATH``, Python imports this at start-up. It
shadows the interpreter's own ``sitecustomize``, which it then runs."""

import os

if os.environ.get("TESTMAP_TEST") and os.environ.get("TESTMAP_OUT"):
    import atexit
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import testmap_core

    testmap_core.start()
    atexit.register(testmap_core.dump, "sub")

import runpy as _runpy
import sysconfig as _sysconfig

_own = os.path.join(_sysconfig.get_paths()["stdlib"], "sitecustomize.py")
if os.path.exists(_own):
    _runpy.run_path(_own, run_name="sitecustomize")
