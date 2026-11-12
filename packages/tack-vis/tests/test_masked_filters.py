# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Filters that leave hidden and refined cells out in place (#405): ``clip``,
``extract_surface``, ``extract_edges``, ``triangulate``, ``feature_edges``,
``glyphs`` and ``surface_normals`` (whose output keeps them, marked), which extracted the other cells first (``ghosts.visible``) -- a copy
of every box below an AMR hierarchy's finest level, its positions stored, its
fields copied and the caches of its topology lost on every call.

Each case runs the filter on the dataset and on ``visible(dataset)`` and
compares the cells as sets: each one's type, its points' positions and values
in its own order, and its cell value. ``clip``, ``extract_surface`` and
``glyphs`` are on the points they use, numbered in the same order;
``extract_edges`` and ``triangulate`` keep the input's points.
"""

import numpy as np
import pytest

import tack
import tack.dataset as td
from tack import arrays
from tack.dataset.ghosts import visible

pytestmark = pytest.mark.quick      # the inner-loop set; see CLAUDE.md, Tests


def _masked(data, seed=2):
    rng = np.random.default_rng(seed)
    marks = np.zeros(data.num_cells, np.uint8)
    marks[rng.random(data.num_cells) < 0.2] = td.Ghosts.HIDDENCELL
    marks[rng.random(data.num_cells) < 0.2] |= td.Ghosts.REFINEDCELL
    marks[rng.random(data.num_cells) < 0.1] |= td.Ghosts.DUPLICATECELL
    data.ghosts = td.Ghosts(cells=tack.array(marks, tack.u8))
    x = arrays.materialize(data.geometry.values).numpy().astype(float)
    dtype = data.dtype
    data.fields["s"] = td.Field(td.H1(data), tack.array(
        (np.sin(2 * x[:, 0]) + x[:, 1] - 0.3 * x[:, 2]).astype(dtype.numpy_dtype)))
    data.fields["v"] = td.Field(td.H1(data), tack.array(np.sin(x).astype(dtype.numpy_dtype),
                                                       tack.vec(3, dtype)))
    data.fields["c"] = td.Field(td.Constant(data), tack.array(
        np.arange(data.num_cells).astype(dtype.numpy_dtype)))
    return data


def _cells(out):
    """Each cell as (type, its points' positions, their values, its cell value),
    sorted."""
    t = out.topology
    offsets, connectivity, types = (np.asarray(arrays.to_host(a)).reshape(-1) for a in
                                    (t.offsets, t.connectivity, t.types))
    if types.size == 1 and out.num_cells > 1:
        types = np.repeat(types, out.num_cells)
    x = arrays.materialize(out.geometry.values).numpy()
    s = out.fields["s"].values.numpy()
    c = out.fields["c"].values.numpy() if "c" in out.fields else np.zeros(out.num_cells)
    return sorted((int(types[i]),
                   tuple(x[p].tobytes() for p in connectivity[offsets[i]:offsets[i + 1]]),
                   tuple(float(s[p]) for p in connectivity[offsets[i]:offsets[i + 1]]),
                   float(c[i])) for i in range(out.num_cells))


def _dtypes():
    from tack.runtime.dispatch import get_backend

    return [tack.f32] + ([tack.f64] if get_backend().supports_f64 else [])


def _solids(dtype):
    return {"image": td.uniform_grid((9, 8, 7), spacing=(0.4, 0.5, 0.3), dtype=dtype),
            "tetrahedra": td.tetrahedralize(td.uniform_grid((7, 6, 5), dtype=dtype)),
            "hexahedra": td.extract_cells(td.uniform_grid((7, 6, 5), dtype=dtype),
                                          np.arange(120))}


@pytest.mark.parametrize("kind", ["image", "tetrahedra", "hexahedra"])
def test_solid_filters_leave_masked_cells_out_as_extracting_them_did(backend, kind):
    for dtype in _dtypes():
        data = _masked(_solids(dtype)[kind])
        rest = visible(data)
        centre = arrays.materialize(data.geometry.values).numpy().astype(float).mean(axis=0)
        plane = td.Plane(centre, (0.3, 0.2, 1.0))
        for name, run in (("clip", lambda d: td.clip(d, "s", 1.0)),
                          ("clip by a plane", lambda d, plane=plane: td.clip(d, plane)),
                          ("extract_surface", td.extract_surface),
                          ("glyphs", lambda d: td.glyphs(d, "v", scale_by="s"))):
            a, b = run(data), run(rest)
            assert (a.num_cells, a.num_points) == (b.num_cells, b.num_points), name
            if name != "glyphs":
                assert _cells(a) == _cells(b), name
        a, b = td.extract_edges(data), td.extract_edges(rest)
        assert a.num_points == data.num_points and _cells(a) == _cells(b)


def test_surface_filters_leave_masked_cells_out_as_extracting_them_did(backend):
    for dtype in _dtypes():
        for sheet in (td.uniform_grid((12, 10), spacing=(0.3, 0.4), dtype=dtype),
                      td.triangulate(td.uniform_grid((12, 10), spacing=(0.3, 0.4),
                                                     dtype=dtype)),
                      td.as_polygons(td.uniform_grid((12, 10), dtype=dtype))):
            data = _masked(sheet)
            rest = visible(data)
            for run in (td.triangulate, lambda d: td.feature_edges(d, angle=10.0)):
                assert _cells(run(data)) == _cells(run(rest))


def test_glyphs_go_only_where_cells_are_left(backend):
    data = _masked(td.uniform_grid((8, 7, 6)))
    a = td.glyphs(data, "v", scale_by="s")
    b = td.glyphs(visible(data), "v", scale_by="s")
    assert np.array_equal(a.geometry.values.numpy(), b.geometry.values.numpy())
    assert np.array_equal(a.topology.connectivity.numpy(), b.topology.connectivity.numpy())
    assert np.array_equal(a.fields["s"].values.numpy(), b.fields["s"].values.numpy())


def test_a_masked_grid_keeps_its_faces_across_surface_calls(backend):
    # Extracting made another mesh, and derived its faces, every call; the
    # topology's own faces are derived once and kept.
    data = _masked(td.uniform_grid((8, 7, 6)))
    first = td.extract_surface(data)
    faces = data.topology._faces
    assert faces is not None
    second = td.extract_surface(data)
    assert data.topology._faces is faces and _cells(first) == _cells(second)


def test_surface_normals_keep_masked_cells_and_leave_them_out_of_the_normals(backend):
    # The output is the input's cells, so hidden and refined cells stay, marked,
    # with a zero normal; the others' normals and winding, and the normals at
    # the points they use, are the extracted cells' to the bit.
    rng = np.random.default_rng(5)
    for dtype in _dtypes():
        flat = td.triangulate(td.uniform_grid((14, 12), spacing=(0.3, 0.4), dtype=dtype))
        x = arrays.materialize(flat.geometry.values).numpy().astype(float)
        x[:, 2] = 0.5 * np.sin(x[:, 0])
        rows = flat.topology.connectivity.numpy().reshape(-1, 3).copy()
        turned = rng.random(len(rows)) < 0.3                 # work for the winding
        rows[turned] = rows[turned][:, ::-1]
        triangles = td.DataSet(td.UnstructuredTopology(
            np.full(len(rows), 5, np.uint8), np.arange(0, 3 * len(rows) + 1, 3),
            rows.reshape(-1)), tack.array(x.astype(dtype.numpy_dtype), tack.vec(3, dtype)))
        for surface in (triangles, td.uniform_grid((14, 12), spacing=(0.3, 0.4), dtype=dtype),
                        td.as_polygons(td.uniform_grid((14, 12), dtype=dtype))):
            data = _masked(surface)
            kept = (data.ghosts.cells.numpy() & (td.Ghosts.HIDDENCELL | td.Ghosts.REFINEDCELL)) == 0
            for weights in (None, "area"):
                a = td.surface_normals(data, weights=weights)
                b = td.surface_normals(visible(data), weights=weights)
                assert a.num_cells == data.num_cells
                assert np.array_equal(a.ghosts.cells.numpy(), data.ghosts.cells.numpy())
                normals = a.fields["cell_normals"].values.numpy()
                assert not normals[~kept].any()
                assert np.array_equal(normals[kept], b.fields["cell_normals"].values.numpy())
                where = {p.tobytes(): i for i, p in
                         enumerate(arrays.materialize(a.geometry.values).numpy())}
                at = [where[p.tobytes()] for p in arrays.materialize(b.geometry.values).numpy()]
                assert np.array_equal(a.fields["normals"].values.numpy()[at],
                                      b.fields["normals"].values.numpy())
