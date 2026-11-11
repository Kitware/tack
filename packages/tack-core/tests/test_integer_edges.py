# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Integers at the 64-bit edges kept as they are (#412, found by a code review):
an implicit array's host operand typed from its least and greatest values
(``abs`` of the i64 minimum wrapped to it, so it became 0 in i32, and 2**63
became -2**63 in i64); ``tack.arange`` of integers made in an integer type
(NumPy's goes through float64 past 2**63, two values rounding to one);
``CountingArray`` in its declared type on the host and in kernels; and a raw
DLPack capsule copied. Values are checked against Python's integers."""

import numpy as np
import pytest

import tack
from tack import arrays
from tack.runtime.dispatch import get_backend

pytestmark = pytest.mark.quick      # the inner-loop set; see CLAUDE.md, Tests


@pytest.mark.parametrize("values, dtype", [
    (np.array([-2**63, 5], np.int64), tack.i64),
    (np.array([2**63, 2**64 - 1], np.uint64), tack.u64),
    (np.array([2**31, -1], np.int64), tack.i64),
    (np.array([-5, 7], np.int64), tack.i32),
    (np.array([3, 4], np.uint32), tack.i32),
])
def test_an_implicit_arrays_integers_keep_their_values(backend, values, dtype):
    view = arrays.View(values, 0, len(values))
    assert view.dtype == dtype
    assert view.numpy().tolist() == [int(v) for v in values]
    assert arrays.materialize(view).numpy().tolist() == [int(v) for v in values]


@pytest.mark.parametrize("args, dtype, expected", [
    ((2**63, 2**63 + 3), None, tack.u64),
    ((2**63, 2**63 + 3), tack.u64, tack.u64),
    ((2**64 - 3, 2**64), None, tack.u64),
    ((-2**63, -2**63 + 3), None, tack.i64),
    ((2**31 - 1, 2**31 + 2), None, tack.i64),
    ((5,), None, tack.i32),
    ((10, 0, -3), None, tack.i32),
    ((0,), None, tack.i32),
])
def test_arange_of_integers_keeps_every_value(backend, args, dtype, expected):
    out = tack.arange(*args, dtype=dtype)
    assert out.dtype == expected
    assert out.numpy().tolist() == list(range(*args))


def test_arange_of_floats_and_refusals_are_as_before(backend):
    assert tack.arange(0, 1, 0.25).numpy().tolist() == [0.0, 0.25, 0.5, 0.75]
    if get_backend().supports_f64:             # Metal has none
        assert tack.arange(3, dtype=tack.f64).dtype == tack.f64
    with pytest.raises(TypeError, match="no integer type"):
        tack.arange(-1, 2**63 + 2**62, 2**62)      # -1 beside 3 * 2**62 - 1


@pytest.mark.parametrize("size, start, step, dtype", [
    (3, 2**63, 1, tack.u64),
    (3, 2**63 + 5, -2, tack.u64),
    (2, 2**63, 1, None),
    (4, 10, -3, None),
    (3, -2**40, -1, tack.i64),
])
def test_counting_arrays_hold_their_type_on_the_host_and_in_kernels(backend, size, start,
                                                                      step, dtype):
    counting = arrays.CountingArray(size, start=start, step=step, dtype=dtype)
    expected = [start + step * k for k in range(size)]
    assert counting.numpy().tolist() == expected
    assert arrays.materialize(counting).numpy().tolist() == expected
    assert isinstance(counting, arrays.CountingArray)


def test_a_raw_dlpack_capsule_copies(backend):
    source = tack.arange(4)
    copied = tack.from_dlpack(source.__dlpack__(), copy=True)
    assert copied.numpy().tolist() == [0, 1, 2, 3]
    tack.copyto(copied, np.full(4, 9, np.int32))
    assert source.numpy().tolist() == [0, 1, 2, 3]            # independent storage
