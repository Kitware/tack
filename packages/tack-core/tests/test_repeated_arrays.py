# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""An array held by several device-class objects is one kernel parameter.

Objects over the same space hold the same arrays, so a kernel reading
several of them used to take each array once per object: a tack-expr
tensor divergence on a quadratic grid passed 289 pointers, past Level
Zero's 2 KB of kernel arguments (#474). The template rewrite now names a
repeat by the first holder's parameter (`template_rewrite.shared_arrays`),
so lowering never sees the duplicate, and the expansion passes it once.
"""

import numpy as np
import pytest

import tack
from tack.lang.ir_bounds import HIDDEN_PARAMS
from tack.runtime import dispatch, kernel_utils
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


@tack.device_class
class Grid:
    def __init__(self, values):
        self.values = values


@tack.device_class
class Twice:
    """One object holding one array under two names, and a child holding it too."""

    def __init__(self, a, b, child):
        self.a, self.b, self.child = a, b, child

    @tack.func
    def get(self, i):
        return self.a[i] + 10.0 * self.b[i] + 100.0 * self.child.values[i]


@tack.device_class
class Points:
    def __init__(self, positions):
        self.positions = positions


@tack.kernel
def combine(a, b, c, out, n):
    for i in range(n):
        out[i] = a.at(i) + 2.0 * b.at(i) + 4.0 * c.at(i)


@tack.kernel
def grid_sum(a, b, out):
    for i, j in tack.ndindex(b.values.shape[0], b.values.shape[1]):
        out[i, j] = a.values[i, j] + 2.0 * b.values[i, j]


@tack.kernel
def scale_into(src, dst, n):
    for i in range(n):
        dst.values[i] = 3.0 * src.values[i]


@tack.kernel
def twice(t, out, n):
    for i in range(n):
        out[i] = t.get(i)


@tack.kernel
def lengths(p, q, out, n):
    for i in range(n):
        out[i] = p.positions[i].norm() + 2.0 * q.positions[i].norm()


@tack.kernel
def mixed(x, g, out, n):
    for i in range(n):
        out[i] = x[i] + 2.0 * g.values[i]


def _variants(kernel):
    return kernel_utils.kernel_cache_slot(get_backend()._cache, kernel).values()


def _array_params(kernel, debug=None):
    """Each variant's array parameters of its own, sorted: the two debug mode adds
    (`ir_bounds.HIDDEN_PARAMS`) left out, so the counts hold under TACK_DEBUG=1;
    ``debug`` True or False keeps only the variants with or without them."""
    counts = []
    for v in _variants(kernel):
        names = {p.name for p in v.ir.params}
        hidden = set(HIDDEN_PARAMS) <= names
        if debug is None or debug == hidden:
            counts.append(sum(1 for p in v.ir.params
                              if p._is_array and p.name not in HIDDEN_PARAMS))
    return sorted(counts)


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

    # Different maps make a variant of their own, with every array; the same
    # signature with another sharing is not the fast path's entry.
    other = Component(tack.array(values[2]), tack.array(np.arange(n, dtype=np.int32)))
    combine(parts[0], parts[1], other, out, n)
    np.testing.assert_allclose(out.numpy(), values[0][d] + 2 * values[1][d] + 4 * values[2],
                               rtol=1e-6)
    assert _array_params(combine) == [5, 6]
    combine(*parts, out, n)
    np.testing.assert_allclose(out.numpy(), want, rtol=1e-6)
    assert _array_params(combine) == [5, 6]


def test_a_launch_shape_read_through_a_repeat(backend):
    # The ndindex extents are copies of the bound's expressions, outside the
    # loop body: they must name the parameter that remains.
    x = tack.array(np.arange(20, dtype=np.float32).reshape(4, 5))
    out = tack.empty((4, 5), tack.f32)
    grid_sum(Grid(x), Grid(x), out)
    np.testing.assert_allclose(out.numpy(), 3 * x.numpy())
    assert _array_params(grid_sum) == [2]


def test_written_through_one_holder_read_through_another(backend):
    n = 16
    x = tack.array(np.arange(n, dtype=np.float32))
    scale_into(Grid(x), Grid(x), n)
    np.testing.assert_allclose(x.numpy(), 3 * np.arange(n))
    scale_into(Grid(x), Grid(x), n)        # the fast path's variant
    np.testing.assert_allclose(x.numpy(), 9 * np.arange(n))
    assert _array_params(scale_into) == [1]


def test_repeats_inside_one_tree(backend):
    n = 8
    x = tack.array(np.arange(n, dtype=np.float32))
    y = tack.array(np.ones(n, dtype=np.float32))
    out = tack.empty((n,), tack.f32)
    twice(Twice(x, x, Grid(x)), out, n)
    np.testing.assert_allclose(out.numpy(), 111 * np.arange(n))
    twice(Twice(x, y, Grid(x)), out, n)
    np.testing.assert_allclose(out.numpy(), 101 * np.arange(n) + 10)
    assert _array_params(twice) == [2, 3]


def test_shared_vector_arrays(backend):
    n = 10
    rng = np.random.default_rng(5)
    pos = rng.random((n, 3)).astype(np.float32)
    for layout in ("aos", "soa"):
        positions = tack.array(pos, dtype=tack.vec3(tack.f32), layout=layout)
        out = tack.empty((n,), tack.f32)
        lengths(Points(positions), Points(positions), out, n)
        np.testing.assert_allclose(out.numpy(), 3 * np.linalg.norm(pos, axis=1), rtol=1e-5)


def test_a_plain_argument_held_by_a_template_too(backend):
    # Only templates' arrays are shared; this one is passed twice, as before.
    n = 8
    x = tack.array(np.arange(n, dtype=np.float32))
    out = tack.empty((n,), tack.f32)
    mixed(x, Grid(x), out, n)
    np.testing.assert_allclose(out.numpy(), 3 * np.arange(n))


def test_debug_mode_shares_them_too(backend):
    n = 32
    dofs = tack.array(np.arange(n, dtype=np.int32))
    values = [tack.array(np.full(n, k + 1, np.float32)) for k in range(3)]
    out = tack.empty((n,), tack.f32)
    saved = dispatch._debug
    tack.set_debug(True)
    try:
        combine(*(Component(v, dofs) for v in values), out, n)
    finally:
        tack.set_debug(saved)
    np.testing.assert_allclose(out.numpy(), np.full(n, 17.0))
    # Five arrays of its own, beside the two hidden ones bounds checks take.
    assert 5 in _array_params(combine, debug=True)


def test_inspect_shows_the_shared_parameters(backend):
    n = 8
    dofs = tack.array(np.arange(n, dtype=np.int32))
    parts = [Component(tack.array(np.ones(n, np.float32)), dofs) for _ in range(3)]
    out = tack.empty((n,), tack.f32)
    ir = tack.inspect(combine, *parts, out, n, mode="ir")
    header = next(line for line in ir.splitlines() if line.lstrip().startswith("Function"))
    assert header.count("__tmpl_") == 4, header      # three values, one dof map
