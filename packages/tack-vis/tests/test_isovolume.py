# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""``isovolume``: the part where a field lies between two values, a clip at each
end; high-order fields through subdivision, their crossings at the field's roots at
both ends."""

import numpy as np
import pytest

import tack
import tack.dataset as td
from tack.dataset.algorithms.isovolume import _RANGE

pytestmark = pytest.mark.quick


@pytest.fixture(autouse=True)
def _f64(f64_backend):
    """Compared in f64, on every backend with it."""


@tack.func
def _ball(x):
    return ((x[0] - 0.5) * (x[0] - 0.5) + (x[1] - 0.5) * (x[1] - 0.5)
            + (x[2] - 0.5) * (x[2] - 0.5))


@tack.func
def _bend(x):
    return [x[0] + 0.1 * x[1] * x[1], x[1] + 0.12 * x[0] * x[0], x[2] + 0.05 * x[0] * x[1]]


def _hexes():
    grid = td.uniform_grid((5, 5, 5), spacing=(0.25, 0.25, 0.25), dtype=tack.f64)
    hexes = td.extract_cells(grid, np.arange(grid.num_cells))
    return td.DataSet(hexes.topology, hexes.geometry)


def _mesh(cells):
    hexes = _hexes()
    data = {"hexahedra": hexes, "tetrahedra": td.tetrahedralize(hexes),
            "polyhedra": td.as_polyhedra(hexes)}[cells]
    data = td.DataSet(data.topology, data.geometry)
    x = data.positions()
    data.fields["x"] = td.Field(td.H1(data), tack.array(x[:, 0].copy()))
    data.fields["c"] = td.Field(td.Constant(data),
                                tack.array(np.arange(data.num_cells, dtype=np.float64)))
    return data


def _volume(out):
    if isinstance(out, td.PartitionedDataSet):
        return sum(td.integrate(p).measure for p in out)
    return td.integrate(out).measure


@pytest.mark.parametrize("cells", ["hexahedra", "tetrahedra", "polyhedra"])
def test_a_linear_field_between_two_values(backend, cells):
    """A slab of x: its volume, the two clips' cells, both ends' points kept, and a
    cell field going to each piece from its cell."""
    data = _mesh(cells)
    out = td.isovolume(data, "x", 0.3, 0.6)
    assert isinstance(out, td.DataSet)
    np.testing.assert_allclose(_volume(out), 0.3, rtol=0, atol=1e-12)
    twice = td.clip(td.clip(data, "x", 0.3), "x", 0.6, inside=True)
    assert out.num_cells == twice.num_cells and out.num_points == twice.num_points
    np.testing.assert_array_equal(out.positions(), twice.positions())
    x = out.fields["x"].values.numpy()
    assert x.min() == pytest.approx(0.3, abs=1e-12) and x.max() == pytest.approx(0.6, abs=1e-12)
    np.testing.assert_array_equal(out.fields["c"].values.numpy(),
                                  twice.fields["c"].values.numpy())


def test_an_implicit_function_and_a_field_not_held(backend):
    """A slab between two offsets of a plane, and a Field the dataset does not hold:
    the range's field never reaches the output."""
    data = _mesh("hexahedra")
    plane = td.Plane((0.0, 0.5, 0.0), (0.0, 1.0, 0.0))
    out = td.isovolume(data, plane, -0.1, 0.15)
    np.testing.assert_allclose(_volume(out), 0.25, rtol=0, atol=1e-12)
    assert set(out.fields) == {"x", "c"}
    y = td.Field(td.H1(data), tack.array(data.positions()[:, 1].copy()))
    out = td.isovolume(data, y, 0.2, 0.7, fields=["c"])
    np.testing.assert_allclose(_volume(out), 0.5, rtol=0, atol=1e-12)
    assert set(out.fields) == {"c"} and _RANGE not in out.fields


def test_what_isovolume_refuses(backend):
    data = _mesh("hexahedra")
    with pytest.raises(ValueError, match="above its upper"):
        td.isovolume(data, "x", 0.6, 0.3)
    data.fields["v"] = data.geometry
    with pytest.raises(TypeError, match="scalar field"):
        td.isovolume(data, "v", 0.0, 1.0)


def _shell_points(part, level):
    """The cut part's points at ``level`` and the ball's analytic value there."""
    x = part.positions()
    u = part.fields["u"].values.numpy()[:len(x)]
    at = np.abs(u - level) < 1e-12
    r = x[at] - 0.5
    return x[at], (r * r).sum(axis=1)


@pytest.mark.parametrize("space", ["H1", "L2"])
@pytest.mark.parametrize("cells", ["hexahedra", "tetrahedra"])
def test_a_shell_of_a_quadratic_field(backend, cells, space):
    """The shell between two spheres of an order-2 field on straight cells, cut at
    subdivision 3: every crossing on a sub-cell's edge is on its sphere at both ends
    (a hexahedron's centroid points, averages of crossings, are not crossings);
    with exact=False they are not. The volume is the shell's, nearer with exact crossings."""
    data = _mesh(cells)
    data.fields["u"] = td.interpolate(data, _ball, getattr(td, space)(data, order=2),
                                      dtype=tack.f64)
    r1, r2 = 0.2, 0.4
    truth = 4 / 3 * np.pi * (r2 ** 3 - r1 ** 3)
    exact = td.isovolume(data, "u", r1 ** 2, r2 ** 2, subdivision=3)
    linear = td.isovolume(data, "u", r1 ** 2, r2 ** 2, subdivision=3, exact=False)
    whole, part = exact.partitions
    assert part.num_cells and isinstance(part, td.DataSet)
    assert abs(_volume(exact) - truth) < abs(_volume(linear) - truth) < 0.01
    for level in (r1 ** 2, r2 ** 2):
        x, value = _shell_points(part, level)
        # A hexahedron's sub-edges are along the axes, two coordinates on the
        # lattice; its centroid points are inside sub-cells.
        crossing = ((np.abs(x * 12 - np.round(x * 12)) < 1e-9).sum(axis=1) == 2
                    if cells == "hexahedra" else np.ones(len(x), bool))
        assert crossing.sum() > 50
        np.testing.assert_allclose(value[crossing], level, rtol=0, atol=1e-12)
        _, value = _shell_points(linear.partitions[1], level)
        assert np.abs(value - level).max() > 1e-4


def test_a_thin_shell_on_curved_cells(backend):
    """A shell thinner than a lattice step, on a bent geometry: second crossings
    between a lattice point and a first crossing on its sub-edge are at the roots
    too, so the cut's points at either end are where the field takes it."""
    data = _mesh("tetrahedra")
    data.geometry = td.interpolate(data, _bend, td.H1(data, order=2),
                                   dtype=tack.vec3(tack.f64))
    data.fields["u"] = td.interpolate(data, _ball, td.H1(data, order=2), dtype=tack.f64)
    lower, upper = 0.3 ** 2, 0.31 ** 2
    part = td.isovolume(data, "u", lower, upper, subdivision=2).partitions[1]
    assert part.num_cells
    for level in (lower, upper):
        x, _ = _shell_points(part, level)
        assert len(x)
        probed = td.probe(data, x).fields["u"].values.numpy()
        np.testing.assert_allclose(probed, level, rtol=0, atol=1e-10)


def test_pieces_cut_their_shared_edges_alike(backend):
    """With a ghost layer, the pieces' owned points of the cut part are the whole's,
    bit for bit, at both ends."""

    def bent(d):
        d.fields["u"] = td.interpolate(d, _ball, td.H1(d, order=2), dtype=tack.f64)
        d.geometry = td.interpolate(d, _bend, td.H1(d, order=2), dtype=tack.vec3(tack.f64))
        return d

    def owned(out):
        part = out.partitions[1]
        x = part.positions()
        marks = part.ghosts.points if part.ghosts is not None else None
        if marks is not None:
            x = x[(marks.numpy() & td.Ghosts.SKIPPED_POINTS) == 0]
        return x

    grid = td.uniform_grid((5, 5, 5), spacing=(0.25, 0.25, 0.25), dtype=tack.f64)
    whole = owned(td.isovolume(bent(td.tetrahedralize(grid)), "u", 0.04, 0.0961))
    pieces = [td.tetrahedralize(p) for p in td.split(grid, (2, 2, 1), ghost_layers=1)]
    mine = np.concatenate([owned(td.isovolume(bent(p), "u", 0.04, 0.0961)) for p in pieces])
    assert len(mine) == len({tuple(x) for x in mine})
    assert {tuple(x) for x in mine} == {tuple(x) for x in whole}
