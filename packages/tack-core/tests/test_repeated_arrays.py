# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""An array passed through several parameters is launched once.

Device-class objects over the same space hold the same arrays, so a kernel
reading several of them used to take each array once per object: a
tack-expr tensor divergence on a quadratic grid passed 289 pointers, past
Level Zero's 2 KB of kernel arguments (#474). Each repeat now reads through
the first parameter that passes the array.
"""

import numpy as np
import pytest

import tack
from tack.runtime import kernel_utils
from tack.runtime.dispatch import get_backend

pytestmark = pytest.mark.quick


@tack.device_class
class Component:
    """One component of a field: its own values, the shared space's map."""

    def __init__(self, values, dofs):
        self.values, self.dofs = values, dofs

    @tack.func
    def at(self, i):
        return self.values[self.dofs[i]]


@tack.kernel
def combine(a, b, c, out, n):
    for i in range(n):
        out[i] = a.at(i) + 2.0 * b.at(i) + 4.0 * c.at(i)


@tack.kernel
def accumulate(x, y, n):
    for i in range(n):
        y[i] += x[i]


def _array_params(kernel):
    variants = kernel_utils.kernel_cache_slot(get_backend()._cache, kernel).values()
    return sorted(sum(1 for p in v.ir.params if p._is_array) for v in variants)


def test_shared_arrays_are_one_parameter(backend):
    n = 64
    rng = np.random.default_rng(3)
    dofs = tack.array(rng.permutation(n).astype(np.int32))
    values = [rng.random(n).astype(np.float32) for _ in range(3)]
    parts = [Component(tack.array(v), dofs) for v in values]
    out = tack.empty((n,), tack.f32)
    combine(*parts, out, n)
    d = dofs.numpy()
    want = values[0][d] + 2 * values[1][d] + 4 * values[2][d]
    np.testing.assert_allclose(out.numpy(), want, rtol=1e-6)
    # Three values, one dof map, the output.
    assert _array_params(combine) == [5]

    # Different maps make a variant of their own, with every array.
    other = Component(tack.array(values[2]), tack.array(np.arange(n, dtype=np.int32)))
    combine(parts[0], parts[1], other, out, n)
    np.testing.assert_allclose(out.numpy(), values[0][d] + 2 * values[1][d] + 4 * values[2], rtol=1e-6)
    assert _array_params(combine) == [5, 6]


def test_an_array_read_and_written_through_two_parameters(backend):
    n = 32
    x = tack.array(np.arange(n, dtype=np.float32))
    accumulate(x, x, n)
    np.testing.assert_allclose(x.numpy(), 2 * np.arange(n))
    accumulate(x, x, n)    # the fast path's variant
    np.testing.assert_allclose(x.numpy(), 4 * np.arange(n))
