# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""``merge_pieces``: a partitioned dataset back as one dataset, each cell and point
once.

Structured pieces that tile a box come back as the box's grid -- ``split``'s
input exactly, coordinates and offset included -- with or without ghost
layers; other pieces as an unstructured dataset whose cells and points are
the whole's. Ghost cells go, hidden cells stay marked, global ids come along.
Across MPI ranks the result is on the root rank (``_merge_pieces_mpi.py``).
"""

import os
import types

import numpy as np
import pytest
from test_ghost_cells import (
    _curvilinear,
    _planar,
    _rectilinear,
    _unstructured,
    _with_fields,
    described,
)

import tack
import tack.dataset as td
from tack.dataset import Ghosts
from tack.runtime import dispatch
from tack.runtime.cpu import CPUBackend
from tack.runtime.dispatch import env_flag

try:
    import vtkmodules.vtkCommonDataModel  # noqa: F401  (registers the pythonic API)
    from vtkmodules.vtkImagingCore import vtkRTAnalyticSource

    from tack.interop.vtk import vtk_to_dataset
except ImportError:
    if env_flag("TACK_REQUIRE_VTK"):
        raise
    vtk = None
else:
    vtk = types.SimpleNamespace()
try:
    from vtkmodules.vtkFiltersParallelDIY2 import vtkRedistributeDataSetFilter
except ImportError:
    vtkRedistributeDataSetFilter = None

needs_diy2 = pytest.mark.skipif(vtk is None or vtkRedistributeDataSetFilter is None,
                                reason="needs VTK's FiltersParallelDIY2 module")

pytestmark = pytest.mark.quick


class _SecondCPU(CPUBackend):
    """A CPU backend posing as another device's: its arrays are not the default's."""

    display_name = "second device"

    @property
    def memory_domain(self):
        return ("cpu", 1)


@pytest.fixture(autouse=True)
def cpu():
    tack.set_default_device("cpu")
    yield
    tack.set_default_device("cpu")


def _marked(data):
    """``data`` with zero marks where it has none, for ``described``."""
    if data.ghosts is None:
        data.ghosts = Ghosts(cells=np.zeros(data.num_cells, np.uint8),
                             points=np.zeros(data.num_points, np.uint8))
    return data


def _is_whole(merged, whole):
    assert isinstance(merged.topology, td.StructuredTopology)
    assert merged.topology.point_dims == whole.topology.point_dims
    assert merged.topology.offset == whole.topology.offset
    assert type(merged.geometry.values) is type(whole.geometry.values)
    np.testing.assert_array_equal(merged.positions(), whole.positions())
    for name in whole.fields:
        np.testing.assert_array_equal(merged.fields[name].values.numpy(),
                                      whole.fields[name].values.numpy(), err_msg=name)
    assert merged.ghosts is None


@pytest.mark.parametrize("make", [lambda: td.wavelet(4), _rectilinear, _curvilinear, _planar],
                         ids=["uniform", "rectilinear", "explicit", "2d"])
@pytest.mark.parametrize("layers", [0, 1, 2])
def test_structured_pieces_merge_into_the_whole_grid(make, layers):
    whole = _with_fields(make())
    _is_whole(td.merge_pieces(td.split(whole, 3, ghost_layers=layers)), whole)
    _is_whole(td.merge_pieces(td.ghost_cells(td.split(whole, 3), layers=max(layers, 1))), whole)


@pytest.mark.parametrize("layers", [0, 1, 2])
def test_unstructured_pieces_merge_into_the_wholes_cells_and_points(layers):
    whole = _with_fields(td.wavelet(4))
    merged = td.merge_pieces(_unstructured(td.split(whole, (2, 2, 2), ghost_layers=layers)))
    assert isinstance(merged.topology, td.UnstructuredTopology)
    assert (merged.num_cells, merged.num_points) == (whole.num_cells, whole.num_points)
    reference = _marked(td.extract_cells(whole, np.arange(whole.num_cells)))
    assert described(_marked(merged)) == described(reference)


@pytest.mark.parametrize("cut", [
    lambda d: td.contour(d, "RTData", 160.0),
    lambda d: td.slice_plane(d, (0.3, 0.1, 0.0), (1.0, 0.5, 0.2)),
    lambda d: td.clip(d, "RTData", 160.0),
], ids=["contour", "slice", "clip"])
def test_pieces_cut_a_shared_edge_to_the_same_bits(cut):
    """Pieces cut the edges they share as the whole does: each crossing from the end
    first by position, which every piece agrees on, not from the lower point id,
    which is a piece's own. From the lower id a piece and its neighbour
    interpolated an f32 field's crossing from opposite ends, the copies differed
    in their last bits (4e-8), and merge_pieces found no owner for hundreds of
    them; so each cut of ghosted pieces merges, exactly, into the whole's. The
    ghosts come from ghost_cells of unstructured pieces, which appends them, so a
    piece's ids are not in the order of its positions (split's are)."""
    pieces = td.ghost_cells(td.PartitionedDataSet(
        [td.tetrahedralize(p) for p in td.split(td.wavelet(8), 4)]))
    merged = td.merge_pieces(td.PartitionedDataSet([cut(p) for p in pieces]))
    whole = cut(td.tetrahedralize(td.wavelet(8)))
    assert (merged.num_cells, merged.num_points) == (whole.num_cells, whole.num_points)

    def rows(x):
        return x[np.lexsort(x.T[::-1])]

    np.testing.assert_array_equal(rows(merged.positions()), rows(whole.positions()))


@needs_diy2
def test_vtks_pieces_merge_into_the_image():
    image = vtkRTAnalyticSource(whole_extent=(-6, 6, -6, 6, -6, 6))()
    pieces = vtk_to_dataset(vtkRedistributeDataSetFilter(
        number_of_partitions=4, preserve_partitions_in_output=True)(image))
    whole = vtk_to_dataset(image)
    merged = td.merge_pieces(pieces, fields=["RTData"])
    reference = _marked(td.extract_cells(whole, np.arange(whole.num_cells), fields=["RTData"]))
    assert described(_marked(merged)) == described(reference)


def test_hidden_cells_global_ids_and_fields():
    whole = _with_fields(td.wavelet(4))
    marks = np.zeros(whole.num_cells, np.uint8)
    marks[::9] = Ghosts.HIDDENCELL
    whole.ghosts = Ghosts(cells=marks)
    whole.global_ids = td.GlobalIds(cells=np.arange(whole.num_cells) + 100,
                                    points=np.arange(whole.num_points) + 7)
    merged = td.merge_pieces(td.split(whole, 4, ghost_layers=1), fields=["c"])
    assert sorted(merged.fields) == ["c"]
    np.testing.assert_array_equal(merged.ghosts.cells.numpy(), marks)
    np.testing.assert_array_equal(merged.global_ids.cells.numpy(), whole.global_ids.cells.numpy())
    np.testing.assert_array_equal(merged.global_ids.points.numpy(),
                                  whole.global_ids.points.numpy())
    # Unstructured pieces keep their hidden cells too (marked here, since
    # extract_cells, as every filter, leaves them out).
    plain = list(_unstructured(td.split(_with_fields(td.wavelet(4)), 4, ghost_layers=1)))
    hidden = 0
    for piece in plain:
        cells = piece.ghosts.cells.numpy().copy()
        owned = np.flatnonzero((cells & Ghosts.DUPLICATECELL) == 0)
        cells[owned[::5]] |= Ghosts.HIDDENCELL
        hidden += len(owned[::5])
        piece.ghosts = Ghosts(cells=cells, points=piece.ghosts.points)
    unstructured = td.merge_pieces(td.PartitionedDataSet(plain))
    assert int(((unstructured.ghosts.cells.numpy() & Ghosts.HIDDENCELL) != 0).sum()) == hidden
    # Structured pieces that do not tile a box merge as unstructured, hidden
    # cells kept.
    loose = [p for g, p in enumerate(td.split(whole, 4)) if g != 1]
    for p in loose:               # the dropped piece owned some points: unmark them
        p.ghosts = Ghosts(cells=p.ghosts.cells)
    merged_loose = td.merge_pieces(td.PartitionedDataSet(loose))
    assert isinstance(merged_loose.topology, td.UnstructuredTopology)
    assert int(((merged_loose.ghosts.cells.numpy() & Ghosts.HIDDENCELL) != 0).sum()) == \
        sum(int(((p.ghosts.cells.numpy() & Ghosts.HIDDENCELL) != 0).sum()) for p in loose)
    with pytest.raises(TypeError, match="PartitionedDataSet"):
        td.merge_pieces([whole])
    with pytest.raises(ValueError, match="root is a rank"):
        td.merge_pieces(td.split(whole, 2), root=1)


def test_pieces_on_two_devices_and_the_result_device():
    whole = _with_fields(td.wavelet(4))
    second = dispatch.as_device(_SecondCPU())
    pieces = td.PartitionedDataSet([p.to_device(second) if g % 2 else p for g, p in
                                    enumerate(td.split(whole, 4, ghost_layers=1))])
    merged = td.merge_pieces(pieces, device=second)
    assert merged.device == second
    _is_whole(merged.to_device("cpu"), whole)


@tack.func
def _cubic(x):
    return x[0] * x[0] * x[1] + 0.5 * x[2] * x[2] * x[2] - x[0] * x[1] * x[2] + 1.0


@tack.func
def _bend(x):
    return [x[0] + 0.1 * x[1] * x[1], x[1] + 0.12 * x[0] * x[0], x[2] + 0.05 * x[0] * x[1]]


def _high_order(data, layout="blocks"):
    """``data`` bent at order 2, with a cubic in H1 of order 3 (Gauss-Lobatto) and in
    L2 of order 3 (equispaced, in ``layout``), and an L2 field of an order per cell."""
    data.geometry = td.interpolate(data, _bend, td.H1(data, order=2),
                                   dtype=tack.vec3(tack.f64))
    data.fields["u"] = td.interpolate(data, _cubic, td.H1(data, order=3), dtype=tack.f64)
    data.fields["w"] = td.interpolate(
        data, _cubic, td.L2(data, order=3, basis="equispaced", layout=layout), dtype=tack.f64)
    orders = np.arange(data.num_cells) % 2
    data.fields["p"] = td.interpolate(data, _cubic, td.L2(data, order=orders), dtype=tack.f64)
    return data


def test_structured_pieces_of_high_order_merge_into_the_whole_grid(f64_backend):
    """split's pieces of a curved grid with fields of higher order, ghosts or not:
    the whole again, every value bit for bit."""
    whole = _high_order(td.uniform_grid((7, 6, 5), spacing=(0.2, 0.2, 0.25), dtype=tack.f64))
    for layers in (0, 1):
        merged = td.merge_pieces(td.split(whole, 4, ghost_layers=layers))
        assert isinstance(merged.topology, td.StructuredTopology)
        assert repr(merged.geometry.space) == repr(whole.geometry.space)
        np.testing.assert_array_equal(merged.geometry.values.numpy(),
                                      whole.geometry.values.numpy())
        for name in whole.fields:
            assert repr(merged.fields[name].space) == repr(whole.fields[name].space), name
            np.testing.assert_array_equal(merged.fields[name].values.numpy(),
                                          whole.fields[name].values.numpy(), err_msg=name)


@pytest.mark.parametrize("cells", ["hexahedra", "tetrahedra", "mixed"])
@pytest.mark.parametrize("layout", ["blocks", "cells"])
def test_unstructured_pieces_of_high_order_merge_at_their_order(f64_backend, cells, layout):
    """Pieces of curved cells (hexahedra, tetrahedra, or hexahedra beside wedges)
    with fields of higher order merge at their orders: each value is the field's at
    the merged mesh's own nodes, and the whole's volume is kept. A field in another
    space on one piece is refused."""
    grid = td.uniform_grid((5, 5, 5), spacing=(0.25, 0.25, 0.25), dtype=tack.f64)
    hexes = td.extract_cells(grid, np.arange(grid.num_cells))
    if cells == "tetrahedra":
        data = td.tetrahedralize(hexes)
    elif cells == "mixed":
        conn = hexes.topology.connectivity.numpy().reshape(-1, 8)
        rows, types = [], []
        for k, h in enumerate(conn):
            if k % 2:
                rows += [h[[0, 1, 3, 4, 5, 7]], h[[1, 2, 3, 5, 6, 7]]]
                types += [td.shapes.WEDGE] * 2
            else:
                rows.append(h)
                types.append(td.shapes.HEXAHEDRON)
        sizes = [len(r) for r in rows]
        topology = td.UnstructuredTopology(
            tack.array(np.array(types, np.uint8)),
            tack.array(np.concatenate(([0], np.cumsum(sizes))).astype(np.int32)),
            tack.array(np.concatenate(rows).astype(np.int32)), num_points=hexes.num_points)
        data = td.DataSet(topology, hexes.positions())
    else:
        data = hexes
    if cells != "hexahedra" and layout == "cells":
        pytest.skip("the cell-fastest layout is for tensor bases")
    whole = _high_order(td.DataSet(data.topology, data.geometry), layout)
    n = whole.num_cells
    parts = [td.extract_cells(whole, np.arange(0, n // 3)),
             td.extract_cells(whole, np.arange(n // 3, n))]
    merged = td.merge_pieces(td.PartitionedDataSet(parts))
    assert (merged.num_cells, merged.num_points) == (whole.num_cells, whole.num_points)
    assert merged.geometry.space.order == 2
    for name in whole.fields:
        field = merged.fields[name]
        assert type(field.space) is type(whole.fields[name].space)
        expected = td.interpolate(merged, _cubic, field.space, dtype=tack.f64)
        np.testing.assert_allclose(field.values.numpy(), expected.values.numpy(), rtol=0,
                                   atol=1e-12, err_msg=name)
    np.testing.assert_allclose(td.integrate(merged, fields=["u"]).measure,
                               td.integrate(whole, fields=["u"]).measure, rtol=1e-13)
    other = td.extract_cells(whole, np.arange(n // 3, n))
    other.fields["u"] = td.interpolate(other, _cubic, td.H1(other, order=2), dtype=tack.f64)
    with pytest.raises(ValueError, match="one space"):
        td.merge_pieces(td.PartitionedDataSet([parts[0], other]))


def test_a_merged_dataset_of_high_order_travels_to_the_root(f64_backend):
    """What moves a merged dataset to the root rank (_described, _rebuilt) keeps a
    curved geometry and fields of higher order, an order per cell too."""
    import importlib

    module = importlib.import_module("tack.dataset.algorithms.merge_pieces")

    whole = _high_order(td.extract_cells(td.uniform_grid((4, 4, 4), dtype=tack.f64),
                                         np.arange(27)))
    description, payload = module._described(whole)
    again = module._rebuilt(description, payload)
    assert repr(again.geometry.space) == repr(whole.geometry.space)
    np.testing.assert_array_equal(again.geometry.values.numpy(), whole.geometry.values.numpy())
    for name in whole.fields:
        assert repr(again.fields[name].space) == repr(whole.fields[name].space), name
        np.testing.assert_array_equal(again.fields[name].values.numpy(),
                                      whole.fields[name].values.numpy(), err_msg=name)


def test_merge_pieces_across_mpi_ranks(mpiexec):
    script = os.path.join(os.path.dirname(__file__), "_merge_pieces_mpi.py")
    for ranks in (2, 3):
        run = mpiexec(ranks, script)
        assert run.returncode == 0 and "ok" in run.stdout, run.stdout + run.stderr


def merged_across(world, structured, high_order=False):
    """This rank's share of six pieces merged onto rank 0: the dataset described,
    or ``None`` on the other ranks. ``high_order``: of a curved grid with fields of
    higher order, its geometry's and fields' values."""
    if high_order:
        whole = _high_order(td.uniform_grid((7, 6, 5), spacing=(0.2, 0.2, 0.25),
                                            dtype=tack.f64))
    else:
        whole = _with_fields(td.wavelet(4))
    every = list(td.split(whole, 6, ghost_layers=1))
    if not structured:
        every = list(_unstructured(every))
    mine = np.array_split(np.arange(len(every)), world.size)[world.rank]
    merged = td.merge_pieces(td.PartitionedDataSet([every[i] for i in mine], world))
    if merged is None:
        return None
    if high_order:
        return (type(merged.topology).__name__, merged.geometry.values.numpy().tolist(),
                {n: merged.fields[n].values.numpy().tolist() for n in sorted(merged.fields)})
    return (type(merged.topology).__name__, described(_marked(merged)))
