# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Contours of expressions evaluated where the contour reads them, with nothing stored."""

import numpy as np
import pytest
import tack_expr as fx

import tack
import tack.dataset as td


def positions(d):
    return np.asarray(d.positions(), dtype=np.float64)


def sorted_positions(d):
    p = np.round(positions(d), 5)
    return p[np.lexsort(p.T)]


@tack.func
def quadratic(x):
    return x[0] * x[0] + x[1] * x[2]


@pytest.fixture
def order2(backend):
    data = td.uniform_grid((7, 6, 5), origin=(-0.6, -0.5, -0.4), spacing=(0.2, 0.2, 0.2))
    data.fields["u"] = td.interpolate(data, quadratic, td.H1(data, order=2))
    return data


def level_error(surface):
    x = positions(surface)
    u = x[:, 0] ** 2 + x[:, 1] * x[:, 2]
    return float(np.abs(u * u + u - 0.3).max())


def test_a_linear_expression_contours_as_its_stored_field(backend):
    data = td.wavelet(10)
    expr = "RTData * 2 + 40 * coordsX"
    lazy = td.contour(data, fx.field(expr), 330.0)
    stored = td.contour(data, fx.evaluate(expr, data, at="points"), 330.0)
    assert (lazy.num_cells, lazy.num_points) == (stored.num_cells, stored.num_points)
    np.testing.assert_array_equal(sorted_positions(lazy), sorted_positions(stored))
    assert set(data.fields) == {"RTData"}
    assert fx.field(expr, data).space.continuous


def test_exact_crossings_are_on_the_level(order2):
    surface = td.contour(order2, fx.field("u * u + u"), 0.3)
    assert surface.num_cells > 0
    assert level_error(surface) < 1e-5


def test_lattice_crossings_approach_the_level_as_the_subdivision_grows(order2):
    coarse = level_error(td.contour(order2, fx.field("u * u + u"), 0.3, exact=False))
    fine = level_error(td.contour(order2, fx.field("u * u + u"), 0.3, subdivision=8, exact=False))
    assert coarse < 2e-2
    assert fine < coarse / 4


def test_derivative_expressions_contour_cell_by_cell(order2):
    g = fx.field("mag(grad(u))", order2)
    assert not g.space.continuous
    surface = td.contour(order2, g, 0.8, subdivision=2)
    assert surface.num_cells > 0
    x = positions(surface)
    gm = np.linalg.norm(np.stack([2 * x[:, 0], x[:, 2], x[:, 1]], 1), axis=1)
    assert float(np.abs(gm - 0.8).max()) < 1e-5
