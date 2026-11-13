# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Cells of images and rectilinear grids found by arithmetic (issue #371).

``CellLocator(data)`` of an image (``UniformCoordinates``) or a rectilinear grid
(``CartesianProduct``) finds a point's candidate cells from its index
coordinates, or by a binary search of each axis, and builds no boxes or bins;
``levels=`` still asks for the bins. Each candidate is tested and ranked as the
bins' candidates are, so the two must agree bit for bit: the same cell for a
point on a shared face or corner, the same treatment of a point just outside
within the tolerance, the same parametric coordinates. Checked here on solid
and sheet grids, rotated and far from the origin, in f32 and f64, through
``find``, ``probe`` and ``streamlines``.
"""

import numpy as np
import pytest

import tack
import tack.dataset as td
from tack.dataset.locator import CellLocator, cached_locator

pytestmark = pytest.mark.quick      # the inner-loop set; see CLAUDE.md, Tests

_ROTATION = (np.array([[np.cos(0.4), -np.sin(0.4), 0.0], [np.sin(0.4), np.cos(0.4), 0.0],
                       [0.0, 0.0, 1.0]])
             @ np.array([[1.0, 0.0, 0.0], [0.0, np.cos(0.3), -np.sin(0.3)],
                         [0.0, np.sin(0.3), np.cos(0.3)]]))


def _grid(kind, dtype):
    rng = np.random.default_rng(3)
    if kind == "image":
        return td.uniform_grid((9, 7, 5), origin=(1.0, -2.0, 0.5), spacing=(0.3, 2.0, 0.05),
                               dtype=dtype)
    if kind == "rotated":
        return td.uniform_grid((6, 7, 5), origin=(1.0, 2.0, 3.0), spacing=(0.5, 0.3, 0.8),
                               direction=_ROTATION, dtype=dtype)
    if kind == "far":
        return td.uniform_grid((6, 5, 4), origin=(1e5, -3e4, 7e3), spacing=(0.25, 0.5, 1.0),
                               dtype=dtype)
    if kind == "sheet":
        return td.uniform_grid((9, 6), spacing=(0.5, 0.25), dtype=dtype)
    axes = [np.cumsum(rng.uniform(lo, hi, n)) for lo, hi, n in
            ((0.05, 1.0, 9), (0.01, 3.0, 6), (0.2, 0.4, 5))]
    if kind == "rectilinear":
        return td.rectilinear_grid(*axes, dtype=dtype)
    return td.rectilinear_grid(*axes[:2], dtype=dtype)          # a rectilinear sheet


def _points(data, n=4000):
    """Points over the grid's box and past it, at its points (shared corners), next
    to them, halfway between two of them (on faces and edges), and off a
    sheet's plane within and past its tolerance."""
    rng = np.random.default_rng(5)
    grid = data.geometry.values.numpy().astype(float)
    lo, hi = grid.min(axis=0), grid.max(axis=0)
    pick = lambda k: grid[rng.integers(0, len(grid), k)]
    out = np.concatenate([rng.uniform(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo), (n, 3)),
                          pick(n // 4), pick(n // 4) + rng.normal(0.0, 1e-4, (n // 4, 3)),
                          (pick(n // 4) + pick(n // 4)) / 2])
    if len(data.topology.point_dims) == 2:
        out[: n // 2, 2] = lo[2] + rng.normal(0.0, 1e-3, n // 2)
    return out.astype(data.dtype.numpy_dtype)


def _same(a, b):
    return np.array_equal(np.asarray(a).view(np.uint8), np.asarray(b).view(np.uint8))


def _dtypes(backend):
    from tack.runtime.dispatch import get_backend

    return [tack.f32] + ([tack.f64] if get_backend().supports_f64 else [])


@pytest.mark.parametrize("kind", ["image", "rotated", "far", "sheet", "rectilinear",
                                  "rectilinear sheet"])
def test_a_structured_locator_finds_what_the_bins_find(backend, kind):
    for dtype in _dtypes(backend):
        data = _grid(kind, dtype)
        points = _points(data)
        arithmetic, bins = CellLocator(data), CellLocator(data, levels=2)
        assert arithmetic.structured is not None and bins.structured is None
        a, b = arithmetic.find(points), bins.find(points)
        assert int((a.cells.numpy() >= 0).sum()) > len(points) // 3
        assert _same(a.cells.numpy(), b.cells.numpy()), (kind, dtype)
        assert _same(a.pcs.numpy(), b.pcs.numpy()), (kind, dtype)
        held_a = arithmetic._find_holding(points)
        held_b = bins._find_holding(points)
        assert _same(held_a[0].numpy(), held_b[0].numpy())
        assert _same(held_a[1].numpy(), held_b[1].numpy())


def test_points_just_outside_are_found_within_the_tolerance_alike(backend):
    # The tolerance is a thousandth of a cell's diagonal: 1.73e-3 here.
    data = td.uniform_grid((5, 5, 5))
    rng = np.random.default_rng(1)
    rows = []
    for d in (1e-4, 1e-3, 1.5e-3, 1.8e-3, 3e-3):
        rows.append(np.c_[np.full(200, -d), rng.uniform(0.0, 4.0, (200, 2))])
    face = np.concatenate(rows)
    points = np.concatenate([face, 4.0 - face, face[:, [1, 0, 2]], face[:, [1, 2, 0]]])
    a = CellLocator(data).find(points.astype(np.float32))
    b = CellLocator(data, levels=2).find(points.astype(np.float32))
    assert _same(a.cells.numpy(), b.cells.numpy()) and _same(a.pcs.numpy(), b.pcs.numpy())
    found = a.cells.numpy() >= 0
    assert found.reshape(4, 5, 200)[:, :3].all() and not found.reshape(4, 5, 200)[:, 3:].any()


def test_a_structured_locator_builds_no_boxes_or_bins(backend):
    data = td.uniform_grid((8, 8, 8))
    locator = CellLocator(data)
    locator.find(np.full((10, 3), 3.5, np.float32))
    assert locator.lo is None and "_cell_bins" not in locator.__dict__
    assert np.allclose(locator.box[0], 0.0) and np.allclose(locator.box[1], 7.0)
    # Asking for the bins (as a test of them does) still makes them.
    offsets, cells = locator.cell_bins()
    assert locator.lo is not None and int(offsets.numpy()[-1]) >= data.num_cells
    assert cached_locator(data).structured is not None


def test_other_grids_keep_the_bins(backend):
    curvilinear = td.uniform_grid((4, 4, 4))
    curvilinear = td.DataSet(curvilinear.topology, tack.array(
        curvilinear.geometry.values.numpy(), tack.vec(3, tack.f32)))
    decreasing = td.rectilinear_grid([3.0, 2.0, 1.0], [0.0, 1.0], [0.0, 1.0])
    line = td.uniform_grid((6,))
    tets = td.tetrahedralize(td.uniform_grid((3, 3, 3)))
    for data in (curvilinear, decreasing, line, tets):
        assert CellLocator(data).structured is None
    with pytest.raises(ValueError):
        CellLocator(td.uniform_grid((3, 3, 3)), levels=3)


@pytest.mark.parametrize("kind", ["image", "rectilinear", "sheet"])
def test_probe_through_either_locator_gives_the_same_values(backend, kind):
    data = _grid(kind, tack.f32)
    n = data.num_points
    data.fields["s"] = td.Field(td.H1(data), tack.array(np.sin(np.arange(n)).astype(np.float32)))
    data.fields["c"] = td.Field(td.Constant(data), tack.array(
        np.arange(data.num_cells, dtype=np.float32)))
    points = _points(data)
    a = td.probe(data, points, locator=CellLocator(data))
    b = td.probe(data, points, locator=CellLocator(data, levels=2))
    for name in ("s", "c", "valid"):
        assert _same(a.fields[name].values.numpy(), b.fields[name].values.numpy()), name


def test_streamlines_through_either_locator_are_the_same(backend):
    data = td.uniform_grid((12, 10, 8), spacing=(0.5, 0.5, 0.5))
    x = data.geometry.values.numpy()
    velocity = np.c_[-(x[:, 1] - 2.25), x[:, 0] - 2.75, 0.1 + 0.0 * x[:, 2]]
    data.fields["v"] = td.Field(td.H1(data), tack.array(velocity.astype(np.float32),
                                                        tack.vec(3, tack.f32)))
    seeds = np.array([[3.0, 2.0, 0.5], [4.0, 2.25, 1.0], [1.0, 1.0, 3.0]], np.float32)
    for neighbors in (False, True):
        arithmetic = CellLocator(data)
        a = td.streamlines(data, "v", seeds, 0.05, 200, locator=arithmetic,
                           neighbors=neighbors)
        assert "_cell_neighbors" not in arithmetic.__dict__      # no walk, no table
        b = td.streamlines(data, "v", seeds, 0.05, 200, locator=CellLocator(data, levels=2),
                           neighbors=neighbors)
        assert a.num_points > 300
        assert _same(a.geometry.values.numpy(), b.geometry.values.numpy())
        assert _same(a.fields["time"].values.numpy(), b.fields["time"].values.numpy())
