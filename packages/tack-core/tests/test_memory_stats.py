# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The count of Tack's own allocations, per device (`tack.runtime.memory_stats`;
the cut filters' memory plan, D1): every array Tack makes is counted where it is
allocated and given back when its buffer is collected, on every backend alike."""

import gc
import threading

import numpy as np
import pytest

import tack
from tack.lang.array import _empty_for_mpi
from tack.runtime import dispatch
from tack.runtime import memory_stats as ms
from tack.runtime.cpu import CPUBackend

pytestmark = pytest.mark.quick


def _settled(device=None):
    gc.collect()
    return ms.live(device)


def test_an_array_is_counted_until_it_is_collected(backend):
    start = _settled()
    a = tack.empty((1000,), tack.f32)
    assert ms.live() - start == 4000
    b = tack.zeros((10, 10), tack.vec3(tack.i32))
    assert ms.live() - start == 4000 + 1200
    del a
    assert _settled() - start == 1200
    del b
    assert _settled() == start


def test_a_view_is_counted_with_its_buffer_once(backend):
    start = _settled()
    a = tack.zeros((100, 4), tack.f32)
    rows = a[10:20]
    flat = a.reshape((400,))
    assert ms.live() - start == 1600
    del a, flat
    assert _settled() - start == 1600          # held while a view of it is
    del rows
    assert _settled() == start


def test_soa_components_are_counted_each(backend):
    start = _settled()
    a = tack.empty((50,), tack.vec3(tack.f32), layout="soa")
    assert ms.live() - start == 3 * 50 * 4
    del a
    assert _settled() == start


def test_arrays_for_mpi_are_counted(backend):
    start = _settled()
    a = _empty_for_mpi((25,), tack.i64, None)
    assert ms.live() - start == 200
    del a
    assert _settled() == start


def test_memory_from_outside_is_not_counted():
    tack.set_default_device("cpu")
    host = np.arange(1000, dtype=np.float32)
    start = _settled()
    shared = tack.from_dlpack(host)
    pointed = tack.array_from_ptr(host.ctypes.data, (1000,), tack.f32)
    assert ms.live() == start
    copied = tack.array(host)
    assert ms.live() - start == 4000
    del shared, pointed, copied


def test_the_peak_is_kept_until_reset(backend):
    _settled()
    base = ms.reset_peak()
    a = tack.empty((2000,), tack.u8)
    b = tack.empty((3000,), tack.u8)
    del a, b
    gc.collect()
    assert ms.peak() - base == 5000
    assert ms.reset_peak() == ms.live()
    assert ms.peak() == ms.live()


def test_a_reused_block_is_counted_while_an_array_holds_it(backend):
    """Metal's pool and HIP's cache keep a dropped array's block for the next array:
    it is counted while an array holds it, not while it waits."""
    start = _settled()
    for _ in range(3):
        a = tack.empty((1 << 19,), tack.f32)            # 2 MiB, pooled where there is one
        assert ms.live() - start == 1 << 21
        del a
        assert _settled() == start


def test_each_device_has_its_own_count():
    class Second(CPUBackend):
        @property
        def memory_domain(self):
            return ("cpu", 1)

    second = dispatch.as_device(Second())
    tack.set_default_device("cpu")
    start, start_second = _settled(), _settled(second)
    a = tack.empty((100,), tack.f32, device=second)
    assert ms.live(second) - start_second == 400
    assert ms.live() == start
    assert ms.live("cpu") == start
    del a
    assert _settled(second) == start_second


def test_threads_allocating_at_once_leave_the_count_as_it_was(backend):
    start = _settled()
    device = tack.device(str(dispatch.as_device(dispatch.get_backend())))

    def churn():
        with device:
            for _ in range(200):
                held = [tack.empty((64,), tack.f32) for _ in range(5)]
                del held

    threads = [threading.Thread(target=churn) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert _settled() == start


def test_the_fixture_measures_one_calls_peak(backend, peak_memory):
    keep = tack.empty((1000,), tack.f32)                # live before: not in the peak

    def work():
        a = tack.empty((1000,), tack.f32)
        b = tack.empty((2000,), tack.f32)
        del a
        return b

    result, peak = peak_memory(work)
    assert peak == 12000
    assert result.shape == (2000,)
    del keep
