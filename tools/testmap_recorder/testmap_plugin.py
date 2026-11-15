# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""``pytest -p testmap_plugin``: record which repository files each test file
runs (``testmap_core``), into ``TESTMAP_OUT``. Setup and teardown count for
the test file, so a module fixture is credited to its file whichever test
builds it. Needs ``tools/testmap_recorder`` on ``PYTHONPATH``, which also lets this
directory's ``sitecustomize`` record subprocesses."""

import os

import pytest
import testmap_core


def pytest_configure(config):
    testmap_core.start()
    testmap_core.state["test"] = None       # collection belongs to no test file
    testmap_core.patch_tack()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    testmap_core.switch(item.nodeid.split("::")[0])
    yield


def pytest_sessionfinish(session):
    testmap_core.dump("pytest-" + os.environ.get("PYTEST_XDIST_WORKER", "main"))
