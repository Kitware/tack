# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Bounds of expression fields: each cell's interval holds every value the expression takes in
it, and a subdivided contour with them is the contour without them."""

import numpy as np
import pytest
import tack_expr as fx
from tack_expr import lazy

import tack
import tack.dataset as td


@tack.func
def f_u(x):
    return tack.cos(4.0 * x[0]) + tack.sin(3.0 * x[1]) * x[2]


@tack.func
def f_t(x):
    return x[0] * x[1] + 0.3 * x[2] * x[2] - 0.2


@tack.kernel
def cell_ranges(cells, u, lo, hi):
    for c in cells:
        a, b = u.value_range(c)
        lo[cells.entity_id(c)] = a
        hi[cells.entity_id(c)] = b


@pytest.fixture
def data(backend):
    d = td.uniform_grid((5, 4, 4), origin=(0.1, 0.2, 0.3), spacing=(0.3, 0.3, 0.25))
    d.fields["u"] = td.interpolate(d, f_u, td.H1(d, order=3))
    d.fields["T"] = td.interpolate(d, f_t, td.H1(d, order=2))
    d.fields["k"] = td.Field(td.Constant(d), tack.array(np.linspace(0.5, 2.0, d.num_cells)))
    return d


EXPRESSIONS = ("u * T + 1", "exp(-u) - T ^ 2", "abs(u) * k - sqrt(abs(T))", "max(u, T) / 2 - min(u, 0.1)",
               "if(T > 0, u, -u) + sin(T)", "(u - 0.2) ^ 3")


@pytest.mark.parametrize("text", EXPRESSIONS)
def test_each_cells_bound_holds_its_values(data, text):
    f = fx.field(text, data)
    lo = tack.zeros((data.num_cells,), data.dtype)
    hi = tack.zeros((data.num_cells,), data.dtype)
    td.for_each(cell_ranges, data, "cells", f, lo, hi)
    lo, hi = lo.numpy(), hi.numpy()
    assert np.all(hi > -1e37) and np.all(lo < 1e37)
    order = 4
    nodes = fx.evaluate(text, data, at=td.L2(data, order=order, layout="blocks")).values.numpy().reshape(data.num_cells, -1)
    slack = 1e-4 * (1.0 + np.abs(nodes).max())
    assert np.all(nodes.min(axis=1) >= lo - slack)
    assert np.all(nodes.max(axis=1) <= hi + slack)
    assert np.mean(hi - lo) < 20 * np.mean(nodes.max(axis=1) - nodes.min(axis=1) + 1e-3)


def test_unbounded_where_there_is_no_rule(data):
    f = fx.field("mag(grad(u))", data)
    lo = tack.zeros((data.num_cells,), data.dtype)
    hi = tack.zeros((data.num_cells,), data.dtype)
    td.for_each(cell_ranges, data, "cells", f, lo, hi)
    assert np.all(lo.numpy() < -1e37) and np.all(hi.numpy() > 1e37)


@pytest.mark.parametrize("text", ("u * T + 1", "exp(-u) - T ^ 2", "abs(u) - sqrt(abs(T))"))
def test_a_contour_with_bounds_is_the_contour_without(data, text, monkeypatch):
    bounded = td.contour(data, fx.field(text, data), 0.4, subdivision=2)
    monkeypatch.setattr(lazy, "_ranges", lambda text, data: None)
    unbounded = td.contour(data, fx.field(text, data), 0.4, subdivision=2)
    assert bounded.num_cells == unbounded.num_cells
    a = np.sort(bounded.positions().round(5).view(f"V{3 * bounded.positions().dtype.itemsize}").ravel())
    b = np.sort(unbounded.positions().round(5).view(f"V{3 * unbounded.positions().dtype.itemsize}").ravel())
    assert np.array_equal(a, b)
