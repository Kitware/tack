# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""``_host_copy``: large host copies split over threads of their own, with
``np.copyto``'s results, overlapping arrays included."""

import numpy as np
import pytest

from tack.lang import array as array_module

pytestmark = pytest.mark.quick


@pytest.fixture(autouse=True)
def eight_threads(monkeypatch):
    """Split whatever the machine's or TACK_CPU_THREADS' count."""
    monkeypatch.setattr(array_module, "_copy_threads", lambda: 8)


N = 8_000_000                               # 64 MB of int64, over _PARALLEL_COPY_BYTES


@pytest.mark.parametrize("shift", [1, -1, 4096])
def test_overlapping_arrays_copy_as_numpy_copies_them(shift):
    """Split, a chunk of the destination could be written before another thread read
    it as source (six wrong entries at chunk boundaries, Berk's review of #438)."""
    x = np.arange(N + abs(shift), dtype=np.int64)
    expected = x.copy()
    if shift > 0:
        np.copyto(expected[shift:], expected[:-shift])
        array_module._host_copy(x[shift:], x[:-shift])
    else:
        np.copyto(expected[:shift], expected[-shift:])
        array_module._host_copy(x[:shift], x[-shift:])
    np.testing.assert_array_equal(x, expected)


def test_a_large_copy_converts_as_numpy_converts():
    src = np.arange(N, dtype=np.int64) * 3 - 7
    dst = np.empty(N, np.int32)
    array_module._host_copy(dst, src)
    np.testing.assert_array_equal(dst, src.astype(np.int32))
