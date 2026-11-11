# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Pyramids of order 3 and up, as L2 fields: Bergot, Cohen and Duruflé's space (VTK's
modal ``G`` pyramid's), its values at the pyramid's lattice, each node's function a
tensor Bernstein polynomial in the collapsed coordinates (#422)."""

from math import comb

import numpy as np
import pytest
from test_pyramid_order2 import _bounds, _linear, _mesh, _samples, _where

import tack
import tack.dataset as td
from tack.dataset import bases

pytestmark = pytest.mark.quick


@pytest.fixture(autouse=True)
def _f64(f64_backend):
    """Compared in f64, on every backend with it."""


@tack.func
def _cubic(x):
    return (1.0 + x[0] * x[0] * x[1] - 0.5 * x[2] * x[2] * x[2] + x[0] * x[1] * x[2]
            + 0.3 * x[1] * x[1])


def _cubic_at(x):
    return (1.0 + x[:, 0] ** 2 * x[:, 1] - 0.5 * x[:, 2] ** 3 + x[:, 0] * x[:, 1] * x[:, 2]
            + 0.3 * x[:, 1] ** 2)


def _cubic_gradient(x):
    return np.stack([2 * x[:, 0] * x[:, 1] + x[:, 1] * x[:, 2],
                     x[:, 0] ** 2 + x[:, 0] * x[:, 2] + 0.6 * x[:, 1],
                     -1.5 * x[:, 2] ** 2 + x[:, 0] * x[:, 1]], axis=1)


@pytest.mark.parametrize("order", [3, 4, 5, 6])
def test_the_nodes_and_their_functions(order):
    """(p + 1)(p + 2)(2p + 3)/6 nodes in layers; each node's function is 1 there and 0
    at the others, the functions sum to 1, and they hold every polynomial of degree p
    of the pyramid (the reference pyramid's points, base [0, 1]^2, apex over its
    centre)."""
    ijk, points = bases.pyramid_nodes(order)
    n = (order + 1) * (order + 2) * (2 * order + 3) // 6
    assert len(points) == n
    coefficients = bases.pyramid_bernstein(order)
    q = order + 1

    def functions(pts):
        x, y, t = pts.T

        def b(u):
            return np.stack([comb(order, i) * u ** i * (1 - u) ** (order - i)
                             for i in range(q)], axis=1)

        tensor = (b(x)[:, None, None, :] * b(y)[:, None, :, None]
                  * b(t)[:, :, None, None]).reshape(len(x), -1)
        return tensor @ coefficients.T

    np.testing.assert_allclose(functions(points), np.eye(n), rtol=0, atol=1e-12)
    pts = np.random.default_rng(0).random((200, 3))
    np.testing.assert_allclose(functions(pts).sum(axis=1), 1.0, rtol=0, atol=1e-12)

    def physical(c):
        x, y, t = c.T
        return np.stack([(1 - t) * x + t / 2, (1 - t) * y + t / 2, t], axis=1)

    rng = np.random.default_rng(1)
    for _ in range(3):
        powers = [(a, b, c) for a in range(order + 1) for b in range(order + 1 - a)
                  for c in range(order + 1 - a - b)]
        weights = rng.normal(size=len(powers))

        def poly(c, powers=powers, weights=weights):
            X = physical(c)
            return sum(w * X[:, 0] ** a * X[:, 1] ** b * X[:, 2] ** c3
                       for w, (a, b, c3) in zip(weights, powers, strict=True))

        np.testing.assert_allclose(functions(pts) @ poly(points), poly(pts), rtol=0,
                                   atol=1e-10)


@pytest.mark.parametrize("kind", ["pyramids", "mixed"])
@pytest.mark.parametrize("order", [3, 4, 5])
def test_l2_holds_a_cubic(backend, kind, order):
    """On pyramids alone and beside hexahedra and wedges: a cubic is itself at random
    points, its integral exact, its gradient at the nodes exact (but at the apex,
    which the gradients take from just below it)."""
    data = _mesh(kind)
    space = td.L2(data, order=order)
    assert space.basis == "equispaced"
    data.fields["u"] = td.interpolate(data, _cubic, space, dtype=tack.f64)
    points = np.random.default_rng(2).random((300, 3)) * 0.98 + 0.01
    np.testing.assert_allclose(td.probe(data, points).fields["u"].values.numpy(),
                               _cubic_at(points), rtol=0, atol=1e-12)
    integral = td.integrate(data, fields=["u"]).values["u"]
    np.testing.assert_allclose(integral, 1 + 1 / 6 - 1 / 8 + 1 / 8 + 0.1, rtol=1e-13)
    gradients = td.gradients(data, "u", at=space).values.numpy().reshape(-1, 3)
    nodes = td.interpolate(data, _where, space, dtype=tack.vec3(tack.f64)).values.numpy()
    apex = np.zeros(len(nodes), bool)
    offsets = space.offsets.numpy()
    pyramids = data.topology.types.numpy() == int(td.shapes.PYRAMID)
    apex[offsets[1:][pyramids] - 1] = True
    np.testing.assert_allclose(gradients[~apex], _cubic_gradient(nodes[~apex]), rtol=0,
                               atol=1e-10)


def test_bounds_hold_the_field(backend):
    """value_range bounds the field over each pyramid from its Bernstein
    coefficients; flatness is zero for a field the corners interpolate."""
    data = _mesh("pyramids", 2)
    space = td.L2(data, order=4)
    data.fields["u"] = td.interpolate(data, _cubic, space, dtype=tack.f64)
    data.fields["w"] = td.interpolate(data, _linear, space, dtype=tack.f64)
    lo, hi, flat_u, flat_w = (tack.empty((data.num_cells,), tack.f64) for _ in range(4))
    for group in data.launch_groups("cells", [data.fields["u"]]):
        _bounds(data.domain_view("cells", group), data.fields["u"].view(group),
                data.fields["w"].view(group), lo, hi, flat_u, flat_w)
    pcs = tack.empty((64,), tack.vec3(tack.f64))
    tack.copyto(pcs, np.random.default_rng(6).uniform(0, 1, (64, 3)))
    samples = tack.empty((data.num_cells * 64,), tack.f64)
    for group in data.launch_groups("cells", [data.fields["u"]]):
        _samples(data.domain_view("cells", group), data.fields["u"].view(group), pcs, samples)
    samples = samples.numpy().reshape(-1, 64)
    assert (samples.min(axis=1) >= lo.numpy() - 1e-12).all()
    assert (samples.max(axis=1) <= hi.numpy() + 1e-12).all()
    assert flat_u.numpy().max() > 1e-3
    np.testing.assert_allclose(flat_w.numpy(), 0, atol=1e-12)


def test_project_and_whole_cells(backend):
    """project from a cubic of order 3 puts it at the pyramid's nodes; extract_cells
    keeps the field as it is."""
    data = _mesh("conforming")
    data.fields["h"] = td.interpolate(data, _cubic, td.L2(data, order=3), dtype=tack.f64)
    space = td.L2(data, order=4)
    projected = td.project(data, "h", space)
    np.testing.assert_allclose(
        projected.values.numpy(),
        td.interpolate(data, _cubic, space, dtype=tack.f64).values.numpy(), rtol=0,
        atol=1e-12)
    data.fields["u"] = projected
    kept = td.extract_cells(data, np.arange(0, data.num_cells, 3))
    assert kept.fields["u"].space == td.L2(kept, order=4)
    points = td.cell_centers(kept).values.numpy()
    np.testing.assert_allclose(td.probe(kept, points).fields["u"].values.numpy(),
                               _cubic_at(points), rtol=0, atol=1e-12)


def test_what_pyramids_above_order_2_refuse(backend):
    """Equispaced nodes only; H1 above order 2; the cuts above level 2 (until pyramid
    subdivision takes them), which subdivision=2 runs."""
    data = _mesh("pyramids", 1)
    with pytest.raises(NotImplementedError, match="equispaced basis only"):
        _ = td.L2(data, order=3, basis="gauss_lobatto").size
    with pytest.raises(NotImplementedError, match="equispaced basis only"):
        _ = td.L2(data, order=3, basis="gauss_legendre").size
    with pytest.raises(NotImplementedError, match="H1 order 3 on Pyramid"):
        _ = td.H1(data, order=3).size
    data.fields["u"] = td.interpolate(data, _cubic, td.L2(data, order=3), dtype=tack.f64)
    with pytest.raises(NotImplementedError, match="level 1 or 2"):
        td.contour(data, "u", [1.2])
    assert td.contour(data, "u", [1.2], subdivision=2).num_cells > 0
