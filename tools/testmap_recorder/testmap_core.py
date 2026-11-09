# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Record which repository files each test file runs: the recorder that the
pytest plugin (``testmap_plugin``) and the subprocess hook (``sitecustomize``)
share. See ``tools/testmap.py``.

Two sources, since Python never runs a kernel's body:

- ``sys.monitoring``'s ``PY_START`` for every Python function that starts
  (functions only: a module or class body runs once, at import, and would be
  credited to whichever test imported it first). Each location is disabled
  after its first hit and enabled again when the test file changes, so the
  cost is one event per function per test file.
- Hooks on ``Kernel.__call__``, ``Kernel._lower`` and
  ``CallBindings.resolve``, which credit a kernel call with the kernel's
  file, its arguments' classes' files, and every file lowering resolved into
  it (inlined ``@tack.func``, template methods), whichever test lowered it.

Environment: ``TESTMAP_ROOT`` (the repository; files outside it are not
recorded), ``TESTMAP_OUT`` (where each process writes its JSON), and
``TESTMAP_TEST`` (the test file running, for subprocesses).
"""

import inspect
import json
import os
import sys
import threading

ROOT = os.path.realpath(os.environ.get("TESTMAP_ROOT", os.getcwd()))
OUT = os.environ.get("TESTMAP_OUT", "")
# sys.monitoring's tool ids: 0 debugger, 1 coverage, 2 profiler, 5 optimizer.
TOOL = 4

used = {}                      # test file -> set of repository-relative files
state = {"test": os.environ.get("TESTMAP_TEST") or None}
_relative = {}


def _rel(path):
    """``path`` relative to the repository, or None outside it."""
    try:
        return _relative[path]
    except KeyError:
        pass
    real = os.path.realpath(path) if path and not path.startswith("<") else ""
    rel = os.path.relpath(real, ROOT) if real.startswith(ROOT + os.sep) else None
    _relative[path] = rel
    return rel


def note(path):
    rel = _rel(path)
    test = state["test"]
    if rel is not None and test is not None:
        used.setdefault(test, set()).add(rel)


def _on_start(code, offset):
    if code.co_flags & inspect.CO_OPTIMIZED:
        note(code.co_filename)
    return sys.monitoring.DISABLE


def start():
    sys.monitoring.use_tool_id(TOOL, "testmap")
    sys.monitoring.register_callback(TOOL, sys.monitoring.events.PY_START, _on_start)
    sys.monitoring.set_events(TOOL, sys.monitoring.events.PY_START)


def switch(test):
    """Credit what runs from now on to ``test``; its subprocesses too."""
    if test != state["test"]:
        state["test"] = test
        os.environ["TESTMAP_TEST"] = test
        sys.monitoring.restart_events()


def dump(name):
    """Write this process's record as ``OUT/<name>-<pid>.json``."""
    if not OUT or not used:
        return
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"{name}-{os.getpid()}.json"), "w") as f:
        json.dump({test: sorted(files) for test, files in used.items()}, f)


def _file_of(value, depth=0):
    """The source file of a function, kernel or class, if it has one."""
    if depth > 4:
        return None
    for attr in ("func", "python_func", "__wrapped__", "__func__"):
        inner = getattr(value, attr, None)
        if inner is not None and inner is not value:
            found = _file_of(inner, depth + 1)
            if found:
                return found
    code = getattr(value, "__code__", None)
    if code is not None:
        return code.co_filename
    if isinstance(value, type):
        try:
            return inspect.getfile(value)
        except (TypeError, OSError):
            return None
    return None


def patch_tack():
    """Credit kernel calls with the files their kernels came from."""
    from tack.lang import call_bindings, kernel

    lowering = threading.local()
    resolve = call_bindings.CallBindings.resolve

    def resolve_noting(self, node):
        value = resolve(self, node)
        files = getattr(lowering, "files", None)
        if files is not None:
            for v in (self.owner, value):
                found = _file_of(v) if v is not None else None
                if found:
                    files.add(found)
        return value

    lower = kernel.Kernel._lower

    def lower_noting(self, *args, **kwargs):
        outer = getattr(lowering, "files", None)
        lowering.files = set()
        try:
            return lower(self, *args, **kwargs)
        finally:
            found, lowering.files = lowering.files, outer
            if outer is not None:
                outer |= found
            templates = kwargs.get("template_args", args[2] if len(args) > 2 else None)
            for t in (templates or {}).values() if isinstance(templates, dict) else ():
                f = _file_of(type(t))
                if f:
                    found.add(f)
            self.__dict__.setdefault("_testmap_files", set()).update(found)

    call = kernel.Kernel.__call__

    def call_noting(self, *args, **kwargs):
        f = _file_of(self.func)
        if f:
            note(f)
        for a in args:
            f = _file_of(type(a))
            if f:
                note(f)
        try:
            return call(self, *args, **kwargs)
        finally:
            for f in self.__dict__.get("_testmap_files", ()):
                note(f)

    call_bindings.CallBindings.resolve = resolve_noting
    kernel.Kernel._lower = lower_noting
    kernel.Kernel.__call__ = call_noting
