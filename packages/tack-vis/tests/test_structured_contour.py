# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""``contour`` of structured grids by flying edges (issue #316).

Images, rectilinear and curvilinear grids of solid cells are contoured by
flying edges (``algorithms._flying_edges``): crossings counted along the grid's
rows, numbered by a scan and written once, with no per-corner arrays and no
sort. Its output must be the general path's up to the order of points and
triangles -- the same triangles of the same cells, the same crossings to the
bit, the same values carried -- so each case here contours a grid both ways
(the general path by switching flying edges off) and compares the triangles
in the general path's order: by cell, then value, then in the case table's.
"""

import numpy as np
import pytest

import tack
import tack.dataset as td
from tack.dataset.algorithms import _flying_edges

pytestmark = pytest.mark.quick      # the inner-loop set; see CLAUDE.md, Tests


def _general(monkeypatch, *args, **kwargs):
    with monkeypatch.context() as m:
        m.setattr(_flying_edges, "serves", lambda data, field: False)
        return td.contour(*args, **kwargs)


def _triangles(out, names):
    """Each triangle's corners' positions and the named fields at its corners
    (point fields) or on it (cell fields), in the general path's order."""
    rows = out.topology.connectivity.numpy().reshape(-1, 3).astype(np.int64)
    cells = out.fields["cell"].values.numpy()
    levels = (out.fields["contour_value"].values.numpy() if "contour_value" in out.fields
              else np.zeros(len(rows)))
    order = np.lexsort((levels, cells))      # stable: the case table's order within
    corners = out.geometry.values.numpy()[rows[order]]
    fields = {}
    for name in names:
        values = out.fields[name].values.numpy()
        fields[name] = values[rows[order]] if out.fields[name].space.on == "points" \
            else values[order]
    return corners, fields


def _same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and np.array_equal(a.view(np.uint8), b.view(np.uint8))


def _check(monkeypatch, data, value, names=("s",)):
    data.fields["cell"] = td.Field(td.Constant(data), tack.array(
        np.arange(data.num_cells, dtype=np.int64), tack.i64))
    names = [*names, "cell"]
    fast = td.contour(data, "s", value)
    slow = _general(monkeypatch, data, "s", value)
    assert fast.num_cells == slow.num_cells and fast.num_points == slow.num_points
    assert fast.num_cells > 0
    a, fa = _triangles(fast, names)
    b, fb = _triangles(slow, names)
    assert _same(a, b)
    for name in names:
        assert _same(fa[name], fb[name]), name
    if np.ndim(value):
        assert _same(np.sort(fast.fields["contour_value"].values.numpy()),
                     np.sort(slow.fields["contour_value"].values.numpy()))
    return fast


def _dtypes():
    from tack.runtime.dispatch import get_backend

    return [tack.f32] + ([tack.f64] if get_backend().supports_f64 else [])


def _with(data, values, dtype, name="s"):
    data.fields[name] = td.Field(td.H1(data), tack.array(np.asarray(values).astype(
        dtype.numpy_dtype)))
    return data


@pytest.mark.parametrize("several", [False, True])
def test_an_image_contours_as_the_general_path_does(backend, monkeypatch, several):
    for dtype in _dtypes():
        data = td.wavelet((-8, 8, -7, 7, -6, 6), dtype=dtype)
        data.fields["s"] = data.fields["RTData"]
        x = data.geometry.values.numpy()
        data.fields["v"] = td.Field(td.H1(data), tack.array(x.astype(dtype.numpy_dtype),
                                                           tack.vec(3, dtype)))
        _check(monkeypatch, data, [100.0, 150.0, 200.0, 250.0] if several else 150.0,
               names=("s", "v"))


def test_crossings_that_land_on_points_are_those_points(backend, monkeypatch):
    # Values rounded to tenths meet the levels exactly at many points: every
    # crossing there is the point, shared by the edges around it, and a
    # triangle two of whose crossings land on one point is dropped.
    for dtype in _dtypes():
        data = td.uniform_grid((9, 7, 6), spacing=(0.5, 0.3, 0.8), dtype=dtype)
        x = data.geometry.values.numpy().astype(float)
        out = _check(monkeypatch, _with(data, np.round(
            np.sin(x[:, 0]) + np.cos(2 * x[:, 1]) + 0.3 * x[:, 2], 1), dtype), [0.5, 1.0, 1.2])
        assert out.num_points < 3 * out.num_cells


def test_cells_with_a_nan_have_no_surface(backend, monkeypatch):
    rng = np.random.default_rng(4)
    data = td.uniform_grid((9, 7, 6))
    x = data.geometry.values.numpy().astype(float)
    values = np.sin(x[:, 0]) + np.cos(x[:, 1]) + 0.2 * x[:, 2]
    values[rng.integers(0, len(values), 20)] = np.nan
    out = _check(monkeypatch, _with(data, values, tack.f32), [0.5, 1.0])
    # Every point is a triangle's: a crossing whose cells all hold a NaN is not kept.
    used = np.unique(out.topology.connectivity.numpy())
    assert len(used) == out.num_points


def test_rectilinear_and_curvilinear_grids_contour_alike(backend, monkeypatch):
    rng = np.random.default_rng(5)
    for dtype in _dtypes():
        axes = [np.cumsum(rng.uniform(0.1, 1.0, n)) for n in (10, 8, 7)]
        grid = td.rectilinear_grid(*axes, dtype=dtype)
        x = grid.geometry.values.numpy().astype(float)
        _check(monkeypatch, _with(grid, x[:, 0] ** 2 + x[:, 1] - x[:, 2], dtype), [5.0, 10.0])
        image = td.uniform_grid((8, 7, 6), dtype=dtype)
        x = image.geometry.values.numpy().astype(float)
        bent = x + 0.2 * np.sin(x[:, [1, 2, 0]])
        curvilinear = td.DataSet(image.topology, tack.array(bent.astype(dtype.numpy_dtype),
                                                            tack.vec(3, dtype)))
        _check(monkeypatch, _with(curvilinear, np.linalg.norm(bent - 3.0, axis=1), dtype),
               [2.0, 3.0])


def test_ghost_marks_are_carried(backend, monkeypatch):
    pieces = td.ghost_cells(td.split(td.wavelet(8), (2, 1, 1)))
    piece = pieces[0]
    piece.fields["s"] = piece.fields["RTData"]
    out = _check(monkeypatch, piece, 150.0)
    general = _general(monkeypatch, piece, "s", 150.0)
    marks = out.ghosts.cells.numpy()
    order = np.lexsort((np.zeros(out.num_cells), out.fields["cell"].values.numpy()))
    reference = general.ghosts.cells.numpy()
    reference_order = np.lexsort((np.zeros(general.num_cells),
                                  general.fields["cell"].values.numpy()))
    assert _same(marks[order], reference[reference_order]) and marks.any()


def test_what_flying_edges_does_not_take_goes_the_general_way(backend):
    image = td.uniform_grid((6, 5, 4))
    x = image.geometry.values.numpy()
    image = _with(image, x[:, 0] + x[:, 1], tack.f32)
    assert _flying_edges.serves(image, image.fields["s"])
    sheet = _with(td.uniform_grid((6, 5)), np.arange(30.0), tack.f32)
    cell_values = td.uniform_grid((6, 5, 4))
    cell_values.fields["s"] = td.Field(td.Constant(cell_values), tack.array(
        np.arange(cell_values.num_cells, dtype=np.float32)))
    tets = td.tetrahedralize(image)
    for data in (sheet, cell_values, tets):
        assert not _flying_edges.serves(data, data.fields["s"])
    # Unmerged points go the general way too: each triangle its own.
    out = td.contour(image, "s", 4.5, merge_points=False)
    assert out.num_points == 3 * out.num_cells
