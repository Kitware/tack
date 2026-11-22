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
def test_ghost_cells_come_with_their_nodes(cells):
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
def test_filters_over_the_grown_pieces_give_the_wholes(cells):
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


def test_a_field_numbered_by_a_table_comes_numbered_by_entity():
    """An H1 field numbered by an element-to-DOF table (its non-corner values
    relabelled at random) grows as merge_pieces merges one: each cell's nodes read
    through the table, the grown piece's field numbered by entity, the values the
    field numbered by entity gives."""
    from tack.dataset.carry import _node_entries

    pieces = _pieces("hexahedra")
    for k, piece in enumerate(pieces):
        entity = piece.fields["h"].space
        index, _, _, total = _node_entries(entity, tack.zeros((entity.size,), tack.f64))
        rows = index.numpy()[:total].reshape(piece.num_cells, -1)
        relabel = np.arange(entity.size)
        relabel[piece.num_points:] = np.random.default_rng(k).permutation(
            relabel[piece.num_points:])
        table = td.H1(piece, order=3, basis="equispaced", dofs=relabel[rows])
        piece.fields["t"] = td.interpolate(piece, _cubic, table, dtype=tack.f64)
    for after in td.ghost_cells(pieces):
        assert after.fields["t"].space.dofs is None
        # Within an ulp: interpolating into the table space evaluates a node by
        # another path.
        np.testing.assert_allclose(after.fields["t"].values.numpy(),
                                   after.fields["h"].values.numpy(), rtol=0, atol=1e-14)


def test_a_node_a_ghost_shares_keeps_the_pieces_value():
    """Pieces holding different values where they meet: each grown piece keeps its
    own value at a node its cells share with a ghost, corners and nodes inside the
    shared face alike, as its points keep theirs (an order-1 field's); a node only
    the ghosts have takes the neighbour's. The last neighbour's used to win."""
    whole = td.uniform_grid((3, 2, 2), dtype=tack.f64)
    pieces = [td.extract_cells(p, np.arange(p.num_cells)) for p in td.split(whole, 2)]
    for k, piece in enumerate(pieces):
        piece.fields["a"] = td.Field(td.H1(piece), tack.array(np.full(piece.num_points, k + 1.0)))
        space = td.H1(piece, order=2)
        piece.fields["b"] = td.Field(space, tack.array(np.full(space.size, k + 1.0)))
    for k, after in enumerate(td.ghost_cells(td.PartitionedDataSet(pieces))):
        points = after.fields["a"].values.numpy()
        nodes = after.fields["b"].values.numpy()
        np.testing.assert_array_equal(nodes[:after.num_points], points)
        # Order 2 on two cells: 5 x 3 x 3 nodes, 27 of them the piece's own.
        assert np.count_nonzero(nodes == k + 1.0) == 27
        assert np.count_nonzero(nodes == 2.0 - k) == 18


def test_what_ghost_cells_of_higher_order_refuse():
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
