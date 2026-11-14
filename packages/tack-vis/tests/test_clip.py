# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Clip by Viskores' case tables, against VTK's vtkTableBasedClipDataSet: cells,
points, measure and orientation on every linear shape, by implicit functions
and by fields, both ways; fields carried onto the new points and pieces."""

import numpy as np
import pytest
from test_dataset_api import _scalars, _two_hexes_and_a_pyramid

import tack
import tack.dataset as td

try:
    import vtkmodules.all as vtk

    from tack.interop.vtk import dataset_to_vtk, vtk_to_dataset
except ImportError:
    vtk = None

needs_vtk = pytest.mark.skipif(vtk is None, reason="needs VTK, the reference")

SHAPES = {"hexahedron": 12, "tetra": 10, "wedge": 13, "pyramid": 14, "voxel": 11,
          "quad": 9, "pixel": 8, "triangle": 5}
FUNCTIONS = {"sphere": lambda: td.Sphere((1.4, 1.6, 0.9), 1.3),
             "plane": lambda: td.Plane((1.3, 1.1, 0.7), (1, 0.4, -0.3))}


def _grid(kind):
    return vtk.vtkCellTypeSource(cell_type=SHAPES[kind],
                                 blocks_dimensions=(3, 3, 2 if SHAPES[kind] >= 10 else 0))()


def _sizes(grid, what):
    return np.asarray(vtk.vtkCellSizeFilter()(grid).cell_data[what])


def _vtk_clip(grid, values, value, invert):
    copy = vtk.vtkUnstructuredGrid()
    copy.DeepCopy(grid)
    copy.point_data["clip"] = np.asarray(values, np.float32)
    copy.GetPointData().SetActiveScalars("clip")
    # Called as a function, the filter returns both its outputs: the kept side first.
    return vtk.vtkTableBasedClipDataSet(value=value, inside_out=invert)(copy)[0]


def _check_like_vtk(out, reference, what):
    assert (out.num_cells, out.num_points) == (reference.number_of_cells,
                                               reference.number_of_points)
    ours = _sizes(dataset_to_vtk(out), what)
    assert (ours > 0).all(), "every piece the right way out"
    assert ours.sum() == pytest.approx(_sizes(reference, what).sum(), rel=1e-5)
    np.testing.assert_array_equal(
        np.bincount(np.asarray(dataset_to_vtk(out).cell_types), minlength=15),
        np.bincount(np.asarray(reference.cell_types), minlength=15))


@needs_vtk
@pytest.mark.parametrize("kind", SHAPES)
@pytest.mark.parametrize("name", FUNCTIONS)
@pytest.mark.parametrize("invert", [False, True])
def test_clip_by_a_function_is_vtks(backend, kind, name, invert):
    grid = _grid(kind)
    data = vtk_to_dataset(grid)
    f = FUNCTIONS[name]()
    values = td.algorithms.implicit_values(data, f).values.numpy()
    out = td.clip(data, f, inside=invert)
    what = "Volume" if SHAPES[kind] >= 10 else "Area"
    _check_like_vtk(out, _vtk_clip(grid, values, 0.0, invert), what)


@needs_vtk
@pytest.mark.parametrize("kind", ["hexahedron", "wedge", "tetra"])
def test_clip_by_a_field_carries_fields(backend, kind):
    """A linear point field is exact at every new point -- edge crossings and the
    centroid points some cases add -- and each piece has its cell's value."""
    grid = _grid(kind)
    data = vtk_to_dataset(grid)
    x = data.positions()
    data.fields["s"] = td.Field(td.H1(data), _scalars(np.sin(x[:, 0]) + x[:, 1] * x[:, 2]))
    linear = lambda p: 2.0 * p[:, 0] - 0.5 * p[:, 1] + 0.25 * p[:, 2]
    data.fields["lin"] = td.Field(td.H1(data), _scalars(linear(x)))
    data.fields["id"] = td.cell_ids(data)
    out = td.clip(data, "s", 1.1)
    reference = _vtk_clip(grid, data.fields["s"].values.numpy(), 1.1, False)
    _check_like_vtk(out, reference, "Volume")
    np.testing.assert_allclose(out.fields["lin"].values.numpy(), linear(out.positions()),
                               atol=1e-5)
    values = out.fields["s"].values.numpy()
    assert values.min() >= 1.1 - 1e-5            # the kept side, edge points at the value
    # Every piece lies inside the cell it came from.
    centers = td.algorithms.cell_centers(data).values.numpy()
    ids = np.asarray(tack.arrays.to_host(out.fields["id"].values)).astype(int)
    pieces = out.positions()[out.topology.connectivity.numpy()]
    offsets = out.topology.offsets.numpy()
    for k in range(0, out.num_cells, 7):
        corners = pieces[offsets[k]:offsets[k + 1]].mean(axis=0)
        assert np.linalg.norm(corners - centers[ids[k]]) < 1.0


def test_clip_of_a_mixed_mesh_keeps_its_volume(backend):
    data = _two_hexes_and_a_pyramid()
    f = td.Plane((1.0, 0.5, 0.5), (1, 0.2, 0.1))
    kept, other = td.clip(data, f), td.clip(data, f, inside=True)
    total = td.algorithms.cell_geometry(data)[0].values.numpy().sum()
    both = sum(td.algorithms.cell_geometry(td.as_polyhedra(part))[0].values.numpy().sum()
               for part in (kept, other))
    assert both == pytest.approx(total, rel=1e-5)


def test_clip_refuses_what_it_cannot_cut(backend):
    data = _two_hexes_and_a_pyramid()
    # Polyhedra are clipped, into polyhedra (test_clip_polyhedra.py); polygons not yet.
    plane = td.Plane((1, 0, 0), (1, 0, 0))
    cut = td.clip(td.as_polyhedra(data), plane)
    assert isinstance(cut.topology, td.PolyhedralTopology)
    np.testing.assert_allclose(td.cell_geometry(cut).measure.values.numpy().sum(),
                               td.cell_geometry(td.clip(data, plane)).measure.values.numpy()
                               .sum(), rtol=1e-5)
    with pytest.raises(NotImplementedError, match="not polygons"):
        td.clip(td.as_polygons(td.uniform_grid((3, 3))), td.Plane((1, 0, 0), (1, 0, 0)))
    data.fields["c"] = td.Field(td.Constant(data), _scalars([1.0, 2.0, 3.0]))
    with pytest.raises(TypeError, match="point data"):
        td.clip(data, "c", 1.5)


def test_clip_by_integer_point_data(backend):
    """Integer values cross a value between points like floating ones: the crossing
    weights are fractions (they were stored as integers, so cut at the corners)."""
    data = td.rectilinear_grid([0.0, 1.0], [0.0, 1.0], [0.0, 1.0])
    x = data.positions()[:, 0]
    for values in (_scalars(2 * x, tack.i32), _scalars(2 * x)):
        out = td.clip(data, td.Field(td.H1(data), values), 1.0)
        np.testing.assert_allclose(np.unique(out.positions()[:, 0]), [0.5, 1.0])
        assert np.isclose(td.cell_geometry(out).measure.values.numpy().sum(), 0.5)


@pytest.mark.parametrize("inside", [False, True])
def test_a_clip_through_mesh_points_keeps_and_drops_whole_cells(backend, inside):
    """A cut meant through a plane of mesh points, which round-off puts on either
    side of the value (a node placed with cos(pi/2), 6e-17 off a plane through
    the axis), keeps the cells on one side whole and drops the others: no piece
    of no volume between them, no point twice (the tables cut a cell through a
    corner a hair on the dropped side, at the corner)."""
    data = td.uniform_grid((5, 3, 3))                    # x from 0 to 4
    d = data.positions()[:, 0] - 2.0
    on = np.flatnonzero(d == 0)
    d[on] = np.array([0.0, 2e-7, -2e-7])[np.arange(len(on)) % 3]    # f32's round-off
    out = td.clip(data, td.Field(td.H1(data), _scalars(d)), 0.0, inside=inside)
    assert out.num_cells == 8
    assert td.cell_geometry(out).measure.values.numpy().sum() == pytest.approx(8.0)
    x = out.positions()
    assert len(np.unique(x, axis=0)) == len(x) == 27
    assert (x[:, 0] <= 2 + 1e-6).all() if inside else (x[:, 0] >= 2 - 1e-6).all()


@pytest.mark.parametrize("inside", [False, True])
def test_a_clip_drops_cells_that_touch_the_value(backend, inside):
    """Two cells along x, the values -1, 0, 1 at x = 0, 1, 2: each cell has a face at
    the value, and the one on the dropped side only touches it, so it is dropped
    whole (the tables kept a piece of it of no volume, on that face). A cell whose
    corners are all at the value is kept, either way."""
    data = td.uniform_grid((3, 2, 2))
    x = data.positions()[:, 0]
    out = td.clip(data, td.Field(td.H1(data), _scalars(x - 1.0)), 0.0, inside=inside)
    assert out.num_cells == 1
    assert td.cell_geometry(out).measure.values.numpy().sum() == pytest.approx(1.0)
    kept = out.positions()[:, 0]
    assert (kept <= 1).all() if inside else (kept >= 1).all()
    flat = td.Field(td.H1(data), _scalars(np.zeros(data.num_points)))
    assert td.clip(data, flat, 0.0, inside=inside).num_cells == 2


def test_clip_refuses_an_l2_geometry(backend):
    """Clip merges its points across cells, one position each: an L2 geometry is
    refused up front, as for a clip by a function."""
    data = td.rectilinear_grid([0.0, 1.0], [0.0, 1.0], [0.0, 1.0])
    field = td.Field(td.H1(data), _scalars(data.positions()[:, 0]))
    data.geometry = td.project(data, data.geometry, td.L2(data))
    with pytest.raises(TypeError, match="H1 geometry"):
        td.clip(data, field, 0.5)


def test_integer_clip_converts_before_subtracting(backend):
    data = td.uniform_grid((3, 2, 2))
    data.fields["cell"] = td.cell_ids(data)
    values = np.where(data.positions()[:, 1] == 0, -2**31, 2**31 - 1).astype(np.int32)
    field = td.Field(td.H1(data), _scalars(values, tack.i32))
    out = td.clip(data, field)
    np.testing.assert_allclose(np.unique(out.positions()[:, 1]), [0.5, 1.0], atol=1e-6)
    assert np.isclose(td.cell_geometry(out).measure.values.numpy().sum(), 1.0)
    # The two cells meet at x=1: each edge crossing is one shared point.
    positions = out.positions()
    assert sum(np.all(np.isclose(positions, [1, 0.5, 0]), axis=1)) == 1
    np.testing.assert_array_equal(field.values.numpy(), values)
