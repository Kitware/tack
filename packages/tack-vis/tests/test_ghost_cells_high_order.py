# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""``ghost_cells`` of pieces with fields of higher order and a curved geometry: each
ghost cell comes with its nodes, placed through the grown piece's own numbering
(#328)."""

import numpy as np
import pytest

import tack
import tack.dataset as td

pytestmark = pytest.mark.quick


@pytest.fixture(autouse=True)
def _f64(f64_backend):
    """Compared in f64, on every backend with it."""


@tack.func
def _cubic(x):
    return x[0] * x[0] * x[1] + 0.5 * x[2] * x[2] * x[2] - x[0] * x[1] * x[2] + 1.0


@tack.func
def _bend(x):
    return [x[0] + 0.1 * x[1] * x[1], x[1] + 0.12 * x[0] * x[0], x[2] + 0.05 * x[0] * x[1]]


def _bent(data):
    """``data`` unstructured, bent at order 2, with a cubic in H1 of order 3, in L2 of
    order 3 and in L2 of an order per cell (0 and 1)."""
    data = td.DataSet(data.topology, data.geometry)
    data.geometry = td.interpolate(data, _bend, td.H1(data, order=2), dtype=tack.vec3(tack.f64))
    data.fields["h"] = td.interpolate(data, _cubic, td.H1(data, order=3, basis="equispaced"),
                                      dtype=tack.f64)
    data.fields["w"] = td.interpolate(data, _cubic, td.L2(data, order=3, basis="equispaced"),
                                      dtype=tack.f64)
    data.fields["p"] = td.interpolate(data, _cubic, td.L2(data, order=np.arange(data.num_cells)
                                                          % 2), dtype=tack.f64)
    return data


def _grid():
    return td.uniform_grid((7, 7, 5), spacing=(1 / 6, 1 / 6, 1 / 4), dtype=tack.f64)


def _pieces(cells):
    out = []
    for piece in td.split(_grid(), (2, 2, 1)):
        u = td.extract_cells(piece, np.arange(piece.num_cells))
        out.append(_bent(td.tetrahedralize(u) if cells == "tetrahedra" else u))
    return td.PartitionedDataSet(out)


def _whole(cells):
    whole = td.extract_cells(_grid(), np.arange(_grid().num_cells))
    return _bent(td.tetrahedralize(whole) if cells == "tetrahedra" else whole)


@pytest.mark.parametrize("cells", ["hexahedra", "tetrahedra"])
def test_ghost_cells_come_with_their_nodes(backend, cells):
    """Each grown piece keeps its curved geometry and its fields at their orders, every
    value the field's at the grown piece's own nodes -- a node a ghost shares with the
    piece's cells is one value -- and an order per cell for the ghosts too."""
    pieces = _pieces(cells)
    grown = td.ghost_cells(pieces)
    for before, after in zip(pieces, grown, strict=True):
        assert after.num_cells > before.num_cells
        assert repr(after.geometry.space) == repr(before.geometry.space)
        for name in ("h", "w", "p"):
            field = after.fields[name]
            assert type(field.space) is type(before.fields[name].space)
            expected = td.interpolate(after, _cubic, field.space, dtype=tack.f64)
            np.testing.assert_array_equal(field.values.numpy(), expected.values.numpy(),
                                          err_msg=name)
        own = after.fields["p"].space.cell_keys().numpy()[:before.num_cells]
        np.testing.assert_array_equal(own, before.fields["p"].space.cell_keys().numpy())


@pytest.mark.parametrize("cells", ["hexahedra", "tetrahedra"])
def test_filters_over_the_grown_pieces_give_the_wholes(backend, cells):
    """With a ghost layer, the pieces' owned contour points of an order-3 field on the
    curved mesh are the whole's, bit for bit, and merging the grown pieces gives the
    whole back, its fields at their orders."""
    grown = td.ghost_cells(_pieces(cells))
    whole = _whole(cells)

    def owned(surface):
        x = surface.positions()
        marks = surface.ghosts.points if surface.ghosts is not None else None
        if marks is not None:
            x = x[(marks.numpy() & td.Ghosts.SKIPPED_POINTS) == 0]
        return {tuple(p) for p in x}

    mine = set()
    for piece in grown:
        mine |= owned(td.contour(piece, "h", [1.1]))
    assert mine == owned(td.contour(whole, "h", [1.1]))
    merged = td.merge_pieces(grown)
    assert (merged.num_cells, merged.num_points) == (whole.num_cells, whole.num_points)
    for name in ("h", "w"):
        expected = td.interpolate(merged, _cubic, merged.fields[name].space, dtype=tack.f64)
        np.testing.assert_allclose(merged.fields[name].values.numpy(),
                                   expected.values.numpy(), rtol=0, atol=1e-13)


def test_what_ghost_cells_of_higher_order_refuse(backend):
    """Structured pieces with fields of higher order; refreshing a field of higher
    order (update_ghosts); an L2 geometry."""
    structured = list(td.split(_grid(), (2, 1, 1)))
    for piece in structured:
        piece.fields["h"] = td.interpolate(piece, _cubic, td.H1(piece, order=2),
                                           dtype=tack.f64)
    with pytest.raises(NotImplementedError, match="unstructured pieces"):
        td.ghost_cells(td.PartitionedDataSet(structured))
    grown = td.ghost_cells(_pieces("hexahedra"))
    with pytest.raises(TypeError, match="comes with its ghost cells"):
        td.update_ghosts(grown, "w")
    loose = [td.shrink(p, 1.0, fields=["h"]) for p in _pieces("hexahedra")]
    with pytest.raises(NotImplementedError, match="H1 geometry"):
        td.ghost_cells(td.PartitionedDataSet(loose))
