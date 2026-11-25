# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""``ghost_cells`` of pieces with fields of higher order and a curved geometry: each
ghost cell comes with its nodes, placed through the grown piece's own numbering,
structured pieces' ghosts from their owners; and ``update_ghosts`` refreshing such
fields (#328)."""

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


def _unowned_lost(piece, name):
    """``piece``'s field ``name`` with every value no cell of its own has set to NaN:
    what ``update_ghosts`` must bring back."""
    from tack.dataset.algorithms._cell_nodes import value_places

    field = piece.fields[name]
    values = field.values.numpy().copy()
    sizes, starts, places = value_places(field.space, field.values)
    own = (piece.ghosts.cells.numpy() & td.Ghosts.DUPLICATECELL) == 0
    kept = np.zeros(len(values), bool)
    for c in np.flatnonzero(own):
        kept[places[starts[c]:starts[c] + sizes[c]]] = True
    values[~kept] = np.nan
    piece.fields[name] = td.Field(field.space, tack.array(values), field.unit)
    return int((~kept).sum())


@pytest.mark.parametrize("layers", [1, 2])
@pytest.mark.parametrize("cells", ["hexahedra", "tetrahedra"])
def test_update_ghosts_refreshes_fields_of_higher_order(cells, layers):
    """update_ghosts brings back every value of an H1 field of order 3, an L2 field and
    an L2 field of an order per cell that no cell of the piece's own has, from their
    owners; a field made after ghost_cells too."""
    grown = td.ghost_cells(_pieces(cells), layers=layers)
    for piece in grown:
        piece.fields["late"] = td.interpolate(piece, _cubic, td.H1(piece, order=2),
                                              dtype=tack.f64)
    for name in ("h", "w", "p", "late"):
        expected = [piece.fields[name].values.numpy() for piece in grown]
        assert sum(_unowned_lost(piece, name) for piece in grown)
        td.update_ghosts(grown, name)
        for piece, values in zip(grown, expected, strict=True):
            np.testing.assert_array_equal(piece.fields[name].values.numpy(), values,
                                          err_msg=name)


def test_update_ghosts_gives_a_shared_node_its_owners_value():
    """A node on the face two pieces share is the lower piece's after update_ghosts, as
    the face's points are (an order-1 field's), and the nodes only the ghosts have are
    their owners'."""
    whole = td.uniform_grid((3, 2, 2), dtype=tack.f64)
    pieces = [td.extract_cells(p, np.arange(p.num_cells)) for p in td.split(whole, 2)]
    grown = td.ghost_cells(td.PartitionedDataSet(pieces))
    for k, piece in enumerate(grown):
        piece.fields["a"] = td.Field(td.H1(piece), tack.array(np.full(piece.num_points, k + 1.0)))
        space = td.H1(piece, order=2)
        piece.fields["b"] = td.Field(space, tack.array(np.full(space.size, k + 1.0)))
    td.update_ghosts(grown, "a", "b")
    for piece in grown:
        points = piece.fields["a"].values.numpy()
        nodes = piece.fields["b"].values.numpy()
        np.testing.assert_array_equal(nodes[:piece.num_points], points)
        # Order 2 on two cells: 5 x 3 x 3 nodes, the 27 of the first cell piece 0's.
        assert (np.count_nonzero(nodes == 1.0), np.count_nonzero(nodes == 2.0)) == (27, 18)


def _structured(curved, split):
    out = []
    for piece in td.split(_grid(), split):
        if curved:
            piece.geometry = td.interpolate(piece, _bend, td.H1(piece, order=2),
                                            dtype=tack.vec3(tack.f64))
        piece.fields["h"] = td.interpolate(piece, _cubic, td.H1(piece, order=3),
                                           dtype=tack.f64)
        piece.fields["w"] = td.interpolate(piece, _cubic, td.L2(piece, order=2),
                                           dtype=tack.f64)
        piece.fields["p"] = td.interpolate(piece, _cubic, td.L2(
            piece, order=np.arange(piece.num_cells) % 2), dtype=tack.f64)
        out.append(piece)
    return td.PartitionedDataSet(out)


@pytest.mark.parametrize("split, layers", [((2, 2, 1), 1), ((3, 1, 2), 2)])
@pytest.mark.parametrize("curved", [False, True])
def test_structured_pieces_with_fields_of_higher_order(curved, split, layers):
    """Structured pieces stay structured with their fields of higher order and a curved
    geometry: each grown block's values the field's at its nodes, the ghosts' from
    their owners; update_ghosts refreshes them."""
    pieces = _structured(curved, split)
    grown = td.ghost_cells(pieces, layers=layers)
    for before, after in zip(pieces, grown, strict=True):
        assert isinstance(after.topology, td.StructuredTopology)
        assert after.num_cells > before.num_cells
        assert repr(after.geometry.space) == repr(before.geometry.space)
        for name in ("h", "w", "p"):
            expected = td.interpolate(after, _cubic, after.fields[name].space, dtype=tack.f64)
            np.testing.assert_allclose(after.fields[name].values.numpy(),
                                       expected.values.numpy(), rtol=0, atol=1e-14,
                                       err_msg=name)
    for name in ("h", "w", "p"):
        expected = [piece.fields[name].values.numpy() for piece in grown]
        for piece in grown:
            _unowned_lost(piece, name)
        td.update_ghosts(grown, name)
        for piece, values in zip(grown, expected, strict=True):
            np.testing.assert_array_equal(piece.fields[name].values.numpy(), values,
                                          err_msg=name)


def refreshed_across(world, structured=False):
    """Six pieces' fields of higher order with their unowned values lost and refreshed,
    over this rank's share: ``{gid: {name: values}}``."""
    from tack import communication as tc

    if structured:
        every = list(_structured(True, (3, 2, 1)))
    else:
        every = []
        for piece in td.split(_grid(), (3, 2, 1)):
            every.append(_bent(td.extract_cells(piece, np.arange(piece.num_cells))))
    share = len(every) // world.size
    mine = td.PartitionedDataSet(every[world.rank * share:(world.rank + 1) * share], world)
    grown = td.ghost_cells(mine)
    for piece in grown:
        for name in ("h", "w", "p"):
            _unowned_lost(piece, name)
    td.update_ghosts(grown, "h", "w", "p")
    blocks = tc.Blocks([p.device for p in grown], world)
    return {g: {name: p.fields[name].values.numpy() for name in ("h", "w", "p")}
            for g, p in zip(blocks.local, grown)}


def test_fields_of_higher_order_across_mpi_ranks(mpiexec):
    import os

    script = os.path.join(os.path.dirname(__file__), "_ghost_cells_high_order_mpi.py")
    for ranks in (2, 3):
        run = mpiexec(ranks, script)
        assert run.returncode == 0 and "ok" in run.stdout, run.stdout + run.stderr


def test_what_ghost_cells_of_higher_order_refuse():
    """An L2 geometry; refreshing a field of higher order on pieces ghost_cells did not
    make, or one in a space that differs between pieces."""
    loose = [td.shrink(p, 1.0, fields=["h"]) for p in _pieces("hexahedra")]
    with pytest.raises(NotImplementedError, match="H1 geometry"):
        td.ghost_cells(td.PartitionedDataSet(loose))
    grown = td.ghost_cells(_pieces("hexahedra"))
    for k, piece in enumerate(grown):
        piece.fields["mixed"] = td.interpolate(piece, _cubic, td.L2(piece, order=k % 2 + 1),
                                               dtype=tack.f64)
    with pytest.raises(ValueError, match="spaces differ"):
        td.update_ghosts(grown, "mixed")
