# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Clip by Viskores' case tables, against VTK's vtkTableBasedClipDataSet: cells,
points, measure and orientation on every linear shape, by implicit functions
and by fields, both ways; fields carried onto the new points and pieces."""

import itertools

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


def _clip_output_bytes(out):
    t = out.topology
    return sum(a.nbytes for a in (t.types, t.offsets, t.connectivity, out.geometry.values,
                                  *(f.values for f in out.fields.values())))


@pytest.mark.parametrize("kind", ["tetrahedra", "hexahedra"])
def test_clip_holds_little_more_than_its_output(backend, peak_memory, kind):
    """The input points cells keep are numbered through a map of one entry per input
    point, and only the crossings are sorted (the cut filters' memory plan, D2):
    with a record and a sort for every kept corner, clip held 4.4 to 5 times its
    output at its peak, where it keeps most of the mesh; now 1.4 to 2.6."""
    grid = td.uniform_grid((41, 41, 41), spacing=(0.025, 0.025, 0.025))
    data = (td.tetrahedralize(grid) if kind == "tetrahedra"
            else td.extract_cells(grid, np.arange(grid.num_cells)))
    data = td.DataSet(data.topology, data.geometry)
    x = data.positions()
    data.fields["p"] = td.Field(td.H1(data), tack.array((x[:, 0] + 0.3 * x[:, 1]).astype(np.float32)))
    td.clip(data, "p", 0.5)                               # compiled outside the measurement
    for value in (0.05, 0.5):
        out, peak = peak_memory(td.clip, data, "p", value)
        assert peak <= 3 * _clip_output_bytes(out), (value, peak / _clip_output_bytes(out))


@pytest.mark.parametrize("kind", ["tetrahedra", "hexahedra"])
def test_clip_in_batches_is_clip_cell_by_cell(backend, monkeypatch, kind):
    """A mesh of one shape is counted and written in batches of cells (a thread's,
    or a workgroup's 256 lanes), nothing kept per cell; the output is what
    counting every cell gives, bit for bit: a last batch short, hidden cells,
    both sides, a function (the cut filters' memory plan, D3). It holds the
    batches to the per-cell form as the batches' change left it; that the output
    is what clip gave before that change was checked once, by the 117 clips of
    #463 hashing alike before and after it (CPU and Metal), not by this test."""
    import importlib

    clip_module = importlib.import_module("tack.dataset.algorithms.clip")
    grid = td.uniform_grid((12, 11, 9), spacing=(0.1, 0.12, 0.15))
    data = (td.tetrahedralize(grid) if kind == "tetrahedra"
            else td.extract_cells(grid, np.arange(grid.num_cells)))
    data = td.DataSet(data.topology, data.geometry)
    assert data.num_cells % 256
    x = data.positions()
    data.fields["p"] = td.Field(td.H1(data), tack.array((np.sin(4 * x[:, 0]) + x[:, 1] * x[:, 2])
                                                         .astype(np.float32)))
    data.fields["c"] = td.Field(td.Constant(data), tack.array(np.arange(data.num_cells,
                                                                          dtype=np.float32)))
    marks = np.zeros(data.num_cells, np.uint8)
    marks[::7] = td.Ghosts.HIDDENCELL
    data.ghosts = td.Ghosts(cells=tack.array(marks))
    center = tuple(x.mean(axis=0))
    for kwargs in ({"by": "p", "value": 0.4}, {"by": "p", "value": 0.4, "inside": True},
                   {"by": td.implicit.Sphere(center=center, radius=0.4)}):
        batched = td.clip(data, **kwargs)
        monkeypatch.setattr(clip_module, "_BATCHES", False)
        cellwise = td.clip(data, **kwargs)
        monkeypatch.setattr(clip_module, "_BATCHES", True)
        a, b = batched.topology, cellwise.topology
        for name in ("types", "offsets", "connectivity"):
            np.testing.assert_array_equal(getattr(a, name).numpy(), getattr(b, name).numpy())
        np.testing.assert_array_equal(batched.positions(), cellwise.positions())
        for name in ("p", "c"):
            np.testing.assert_array_equal(batched.fields[name].values.numpy(),
                                          cellwise.fields[name].values.numpy())


# Planes through mesh points of _grid's blocks (integer corners), oblique to the cells.
CORNER_PLANES = {"x+y+z=3": ((1, 1, 1), 3.0), "x+2y-z=2": ((1, 2, -1), 2.0),
                 "2x-y+z=1": ((2, -1, 1), 1.0)}


def _closed_area(grid, normal, c, inside):
    """The surface of the blocks' part on the kept side of a plane: their surface
    clipped by it, and the cut."""
    n = np.asarray(normal, float)
    plane = vtk.vtkPlane(origin=tuple(n * c / n.dot(n)), normal=tuple(-n if inside else n))
    surface = vtk.vtkTriangleFilter()(vtk.vtkDataSetSurfaceFilter()(grid))
    kept = vtk.vtkClipPolyData(clip_function=plane)(surface)[0]
    cut = vtk.vtkCutter(cut_function=plane)(grid)
    return sum(vtk.vtkMassProperties(input_data=vtk.vtkTriangleFilter()(p)).GetSurfaceArea()
               for p in (kept, cut))


@needs_vtk
@pytest.mark.parametrize("kind", SHAPES)
@pytest.mark.parametrize("plane", CORNER_PLANES)
@pytest.mark.parametrize("inside", [False, True])
def test_a_clip_through_mesh_points_oblique_to_the_cells_is_exact(backend, kind, plane, inside):
    """A plane through mesh points but oblique to the cells cuts each edge from such a
    point at the point itself (#309): no crossing on top of a corner, no piece of no
    volume, and no cell naming a point twice but a hexahedral piece that loses an
    edge (no shape: it stays a hexahedron, a corner named twice). VTK's clip, of the
    same volume, has points on top of corners and pieces of no volume (on the
    hexahedra 17 and 14; with inside_out, 23, 13 and 5 inside out)."""
    grid = _grid(kind)
    data = vtk_to_dataset(grid)
    normal, c = CORNER_PLANES[plane]
    values = data.positions() @ np.asarray(normal, float) - c
    out = td.clip(data, td.Field(td.H1(data), _scalars(values)), 0.0, inside=inside)
    out.topology.validate()
    x = out.positions()
    assert len(np.unique(x, axis=0)) == len(x)
    offsets, connectivity = (a.numpy() for a in (out.topology.offsets,
                                                 out.topology.connectivity))
    types = out.topology.types.numpy()
    twice = [len(set(connectivity[a:b])) < b - a for a, b in itertools.pairwise(offsets)]
    assert (types[twice] == 12).all()
    measure = td.cell_geometry(out).measure.values.numpy()
    assert (measure > 1e-6).all()
    what = "Volume" if SHAPES[kind] >= 10 else "Area"
    reference = _sizes(_vtk_clip(grid, values, 0.0, inside), what).sum()
    assert measure.sum() == pytest.approx(reference, rel=1e-5)
    if SHAPES[kind] >= 10 and not any(twice):
        # No face is left inside: the surface is the kept part's. (A collapsed
        # hexahedron's collapsed face names a point twice; the faces derived from
        # the topology do not match it with its neighbour's triangle.)
        area = td.cell_geometry(td.extract_surface(out)).measure.values.numpy().sum()
        assert area == pytest.approx(_closed_area(grid, normal, c, inside), rel=1e-5)


@needs_vtk
@pytest.mark.parametrize("at", ["one corner", "two opposite corners", "three corners"])
@pytest.mark.parametrize("inside", [False, True])
def test_a_hexahedral_piece_that_loses_edges_stays_a_hexahedron(backend, at, inside):
    """A hexahedron whose bottom corners are kept, some of them at the value, keeps a
    piece of its own shape whose edges up from those corners have collapsed: no
    shape, so a hexahedron naming those corners twice, as VTK writes a collapsed
    one. One corner is a plane through it; two opposite corners or three only a
    field that is not linear across the cell gives. The other side has the rest,
    and both are VTK's volumes (VTK's clip, too, counts a corner at the value as
    above it)."""
    grid = vtk.vtkCellTypeSource(cell_type=SHAPES["hexahedron"], blocks_dimensions=(1, 1, 1))()
    data = vtk_to_dataset(grid)
    x, y, z = data.positions().T
    bottom = {"one corner": np.where((x == 0) & (y == 0), 0.0, 1.0 + x + y),
              "two opposite corners": np.where(x == y, 0.0, 1.0 + x),
              "three corners": np.where((x == 1) & (y == 1), 1.5, 0.0)}[at]
    values = np.where(z == 0, bottom, -1.0 - x - 0.5 * y)
    out = td.clip(data, td.Field(td.H1(data), _scalars(values)), 0.0, inside=inside)
    out.topology.validate()
    distinct = {"one corner": 7, "two opposite corners": 6, "three corners": 5}[at]
    if not inside:
        assert out.topology.types.numpy().tolist() == [12]
        assert len(set(out.topology.connectivity.numpy())) == distinct
    assert out.num_points == len(np.unique(out.positions(), axis=0))
    reference = _vtk_clip(grid, values, 0.0, inside)
    assert td.cell_geometry(out).measure.values.numpy().sum() == pytest.approx(
        _sizes(reference, "Volume").sum(), rel=1e-5)


def _case_counts(data, at):
    """One case's outputs of each kind, as ``clip._cell_counts`` counts them: cells,
    connectivity, crossings, centroids, centroid members."""
    counts = [0] * 5
    shapes_left = data[at]
    at += 1
    for _ in range(shapes_left):
        kind, count = data[at], data[at + 1]
        at += 2
        entries = data[at:at + count]
        if kind == 0:
            counts[3] += 1
            counts[4] += count
            counts[2] += sum(1 for p in entries if p >= 8)
        else:
            counts[0] += 1
            counts[1] += count
            counts[2] += sum(1 for p in entries if p >= 8 and p != 20)
        at += count
    return counts


def test_every_case_stays_within_the_bound_the_batches_rest_on():
    """A batch keeps each cell's counts in u8 and its starts within the batch in
    u16, which rest on ``_MOST``, the most entries of any kind one cell makes (49 in
    VisIt's tables): a table that broke the bound would wrap silently, so every
    record a cell can take is walked here, both sides: each case's, and each
    snapped record of a case with corners at the value (``_clip_snap_tables``)."""
    import importlib

    from tack.dataset import _clip_tables as tables

    clip_module = importlib.import_module("tack.dataset.algorithms.clip")
    most = clip_module._MOST
    assert most < 2**8 and (clip_module._OFFSET_BATCH - 1) * most < 2**16
    largest = 0
    for invert in (0, 1):
        data = clip_module._data(invert).tolist()
        for table, cases in tables.CASES[invert].items():
            n = len(cases).bit_length() - 1
            records = clip_module._cases(invert, table, n).tolist()
            for entry, at in enumerate(records):
                if at < 0:
                    continue                            # no snapped record: the case's own
                counts = _case_counts(data, at)
                assert max(counts) <= most, (invert, table, entry, counts)
                largest = max(largest, *counts)
    assert largest <= most and 8 <= most            # a cell kept whole: 8 corners at most


# Clips with no corner near the value, as main's clip gave them before the records
# for corners at the value: sha256 of types, offsets and connectivity (as i64).
_UNCHANGED = {
    ("tetrahedra", False): (3570, "49214f57084bdc6546f3bdadeb6a72fcd91b7ce599b8f3dffdfd7f59da8b9752"),
    ("tetrahedra", True): (1342, "85339becf14b9543c8a2ca267d9b5a2dfc20b0af43b1e9c6be72ec297291e23e"),
    ("hexahedra", False): (1125, "ac6f4f6572c78b4be81924553439b3b05f0548757285f9b347ab310de6b0fd95"),
    ("hexahedra", True): (665, "fa314ec7ed226d77ec511e9ba008bb718362ce0d10062ba59c01b799bfb7591e"),
}


@pytest.mark.parametrize("kind", ["tetrahedra", "hexahedra"])
@pytest.mark.parametrize("inside", [False, True])
def test_a_clip_with_no_corner_near_the_value_is_unchanged(backend, kind, inside):
    """Every cell's case goes through the lookup of corners at the value; with none
    near it (the nearest 7e-4 off), the cells are what the case tables alone made,
    as stored from the clip before that lookup."""
    import hashlib

    grid = td.uniform_grid((12, 11, 9), spacing=(0.1, 0.12, 0.15))
    data = (td.tetrahedralize(grid) if kind == "tetrahedra"
            else td.extract_cells(grid, np.arange(grid.num_cells)))
    data = td.DataSet(data.topology, data.geometry)
    x = data.positions()
    p = (x[:, 0] + 0.3 * x[:, 1] + 0.17 * x[:, 2] ** 2).astype(np.float32)
    assert np.abs(p - np.float32(0.537)).min() > 5e-4
    data.fields["p"] = td.Field(td.H1(data), tack.array(p))
    out = td.clip(data, "p", 0.537, inside)
    digest = hashlib.sha256()
    for ids in (out.topology.types, out.topology.offsets, out.topology.connectivity):
        digest.update(np.ascontiguousarray(ids.numpy().astype(np.int64)).tobytes())
    assert (out.num_cells, digest.hexdigest()) == _UNCHANGED[kind, inside]


def test_the_snap_tables_are_the_generators(tmp_path):
    """_clip_snap_tables.py is make_clip_snap_tables.py's output: regenerated and
    compared decoded, not as bytes (zlib-ng compresses the same tables to other
    bytes)."""
    import importlib.util
    import pathlib
    import subprocess
    import sys

    pytest.importorskip("scipy")
    from tack.dataset import _clip_snap_tables as kept
    from tack.dataset import _clip_tables as tables

    package = pathlib.Path(__file__).resolve().parents[1]
    target = tmp_path / "_clip_snap_tables.py"
    made = subprocess.run([sys.executable, str(package / "tools" / "make_clip_snap_tables.py"),
                           str(target)], capture_output=True, text=True)
    assert made.returncode == 0, made.stderr
    spec = importlib.util.spec_from_file_location("_made_snap_tables", target)
    fresh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fresh)
    for invert in (False, True):
        np.testing.assert_array_equal(fresh.data(invert), kept.data(invert))
        for table in tables.CASES[invert]:
            np.testing.assert_array_equal(fresh.cases(invert, table), kept.cases(invert, table),
                                          err_msg=f"invert {invert}, table {table}")


@pytest.mark.parametrize("dtype", ["f32", "f64"])
def test_the_band_at_the_value_follows_the_values_precision(dtype):
    """A corner is at the value within a millionth of its cell's span in f32 and
    1e-12 in f64: round-off at the nodes (half an ulp to either side of a plane of
    them) still gives whole cells, and in f64 an offset of 3e-8 on cells 0.1 wide
    is a cut, where clip and contour agree."""
    element, numpy_type = (tack.f64, np.float64) if dtype == "f64" else (tack.f32, np.float32)
    grid = td.uniform_grid((11, 3, 3), spacing=(0.1, 0.1, 0.1), dtype=element)
    data = td.extract_cells(grid, np.arange(grid.num_cells))
    x = data.positions()[:, 0].astype(numpy_type)
    on = np.isclose(x, 0.5)
    rounded = x.copy()
    rounded[on] += np.where(np.arange(on.sum()) % 2, 1, -1) * np.spacing(numpy_type(0.5)) / 2
    data.fields["x"] = td.Field(td.H1(data), tack.array(rounded))
    whole = td.clip(data, "x", 0.5)
    assert whole.num_cells == 20 and whole.positions()[:, 0].min() == 0.5
    assert whole.num_points == len(np.unique(whole.positions(), axis=0))
    if dtype == "f64":
        data.fields["x"] = td.Field(td.H1(data), tack.array(x))
        value = 0.5 + 3e-8
        cut = td.clip(data, "x", value).positions()[:, 0].min()
        surface = td.contour(data, "x", value).positions()[:, 0].min()
        assert cut == pytest.approx(value, rel=0, abs=1e-15)
        assert cut == pytest.approx(surface, rel=0, abs=1e-15)
