# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Ghost layers for structured pieces, kept structured: ``ghost_cells``' path for
pieces that are all ``StructuredTopology``.

Each piece is a block of a larger grid: its cells start at ``topology.offset``
there. The pieces must tile a box of cells (VTK's image pieces and ``split``'s
do). A piece becomes the block grown by ``layers`` cells on every side, as far
as the box goes, with a new offset. Everything else is arithmetic on global
indices: a cell belongs to the piece whose block holds it, a point to the
lowest piece whose block holds it (``split``'s rule, and VTK's), and a
piece's ghosts take their values from those owners through two
``tack.communication.Plan``s -- the same ones ``update_ghosts`` runs later.
Uniform coordinates move their origin; rectilinear ones take the axes every
piece gives; explicit positions travel as a point field.

Fields of higher order (``H1`` above order 1, ``L2``) and a curved geometry are
remade on the grown block, each piece's own cells' nodes placed there, and their
ghosts filled from their owners through ``_node_plans``: a value's owner is the
lowest piece with a cell of its own that has it, so a node on a face two blocks
share is the lower block's, as the face's points are.
"""

import math

import numpy as np

import tack
from tack import arrays
from tack.arrays import CartesianProduct, UniformCoordinates
from tack.communication._kernels import scatter
from tack.dataset import _matching
from tack.dataset._matching import INDEX_BITS
from tack.dataset.algorithms._cell_nodes import add_nodes, field_kind, nodes_placed, space_spec
from tack.dataset.algorithms._node_plans import GEOMETRY, refresh
from tack.dataset.algorithms.split import _block_geometry
from tack.dataset.dataset import DataSet, Field, field_label
from tack.dataset.ghosts import Ghosts
from tack.dataset.global_ids import GlobalIds
from tack.dataset.sets import Set
from tack.dataset.spaces import H1, L2, Constant, Values
from tack.dataset.topology import StructuredTopology

__all__ = ["structured_ghost_cells"]


def structured_ghost_cells(blocks, pieces, layers, names):
    """The pieces grown by ``layers`` cells, and the plans of their ghosts:
    ``(pieces, {"points": Plan, "cells": Plan, ...})``, with what ``_node_plans``
    makes the plans of fields of higher order from."""
    kind_of = field_kind
    dims = {len(p.topology.point_dims) for p in pieces}
    every = blocks.all_gather({g: (tuple(p.topology.offset),
                                   tuple(d - 1 for d in p.topology.point_dims))
                               for g, p in zip(blocks.local, pieces)})
    if len({len(o) for o, _ in every} | dims) > 1:
        raise ValueError("ghost_cells needs structured pieces of one dimension")
    lower = np.array([o for o, _ in every], np.int64)
    upper = lower + np.array([c for _, c in every], np.int64)
    whole_lower, whole_upper = lower.min(axis=0), upper.max(axis=0)
    if int(np.prod(upper - lower, axis=1).sum()) != int(np.prod(whole_upper - whole_lower)) \
            or _overlap(lower, upper):
        raise ValueError("ghost_cells needs structured pieces that tile a box of cells "
                         "(their offsets and sizes, as split and VTK's image pieces give)")
    grown_lower = np.maximum(lower - layers, whole_lower)
    grown_upper = np.minimum(upper + layers, whole_upper)

    def flat(local, extent):
        # x fastest
        index = np.zeros(len(local), np.int64)
        for axis in reversed(range(local.shape[1])):
            index = index * extent[axis] + local[:, axis]
        return index

    def grid(lo, hi):
        """The global indices from ``lo`` to ``hi`` (inclusive), x fastest."""
        axes = [np.arange(a, b + 1) for a, b in zip(lo, hi)]
        mesh = np.meshgrid(*axes, indexing="ij")
        return np.stack([m.ravel(order="F") for m in mesh], axis=1).astype(np.int64)

    def point_owner(points):
        owner = np.full(len(points), -1, np.int64)
        for d in range(blocks.count):
            inside = np.all((points >= lower[d]) & (points <= upper[d]), axis=1)
            owner[inside & (owner < 0)] = d
        return owner

    def cell_owner(cells):
        owner = np.full(len(cells), -1, np.int64)
        for d in range(blocks.count):
            owner[np.all((cells >= lower[d]) & (cells < upper[d]), axis=1)] = d
        return owner

    receive = {"points": {}, "cells": {}}
    layout, identities = {}, {}
    for g in blocks.local:
        point_extent = grown_upper[g] - grown_lower[g] + 1
        cell_extent = grown_upper[g] - grown_lower[g]
        points = grid(grown_lower[g], grown_upper[g])
        cells = grid(grown_lower[g], grown_upper[g] - 1)
        p_owner, c_owner = point_owner(points), cell_owner(cells)
        own_points = np.flatnonzero(np.all((points >= lower[g]) & (points <= upper[g]), axis=1))
        own_cells = np.flatnonzero(c_owner == g)
        identity = {}
        for kind, entities, owner in (("points", points, p_owner), ("cells", cells, c_owner)):
            receive[kind][g] = {}
            identity[kind] = np.zeros(len(entities), np.int64)
            for d in np.unique(owner).tolist():
                rows = np.flatnonzero(owner == d)
                extent = (grown_upper[d] - grown_lower[d] + (1 if kind == "points" else 0))
                theirs = flat(entities[rows] - grown_lower[d], extent)
                # (owner, index there), as ghost_cells' unstructured pieces number them.
                identity[kind][rows] = (np.int64(d) << INDEX_BITS) | theirs
                if d != g:
                    receive[kind][g][int(d)] = (theirs, rows)
        layout[g] = (point_extent, cell_extent, p_owner, c_owner, own_points, own_cells)
        identities[g] = (identity["points"], identity["cells"], c_owner == g)
    plans = {kind: _matching.plan(blocks, receive[kind]) for kind in ("points", "cells")}

    rectilinear = _axes(blocks, pieces, lower)
    grown = {}
    for g, piece in zip(blocks.local, pieces):
        point_extent, cell_extent, p_owner, c_owner, own_points, own_cells = layout[g]
        np_, nc = int(np.prod(point_extent)), int(np.prod(cell_extent))
        with blocks.device(g):
            topology = StructuredTopology(tuple(int(d) for d in point_extent),
                                          id_dtype=piece.topology.id_dtype,
                                          offset=tuple(int(v) for v in grown_lower[g]))
            spread = {}                       # the arrays the plans fill, per kind
            point_index = tack.array(own_points, tack.i64)
            cell_index = tack.array(own_cells, tack.i64)

            def placed(values, n, index):
                values = arrays.materialize(values)
                out = tack.empty((n,), values.dtype)
                scatter(values, index, out)
                return out

            for name in names:
                field = piece.fields[name]
                kind = kind_of(field)
                if kind == "points":
                    spread.setdefault("points", {})[field_label(name)] = placed(field.values, np_,
                                                                              point_index)
                elif kind == "cells":
                    spread.setdefault("cells", {})[field_label(name)] = placed(field.values, nc,
                                                                             cell_index)
                elif kind == "blocks" and field.space.varies:
                    # An order per cell: the ghosts' from their owners.
                    spread.setdefault("cells", {})["\0order " + field_label(name)] = placed(
                        field.space.cell_keys(), nc, cell_index)
            marks = piece.ghosts
            spread.setdefault("cells", {})["\0marks"] = placed(
                tack.zeros((piece.num_cells,), tack.u8) if marks is None or marks.cells is None
                else marks.cells, nc, cell_index)
            spread.setdefault("points", {})["\0marks"] = placed(
                tack.zeros((piece.num_points,), tack.u8) if marks is None or marks.points is None
                else marks.points, np_, point_index)
            ids = piece.global_ids
            for kind, index, n in (("points", point_index, np_), ("cells", cell_index, nc)):
                if ids is not None and getattr(ids, kind) is not None:
                    spread[kind]["\0ids"] = placed(getattr(ids, kind), n, index)
            geometry = piece.geometry.values
            implicit = isinstance(geometry, (UniformCoordinates, CartesianProduct))
            if not implicit:
                spread["points"]["\0positions"] = placed(
                    _point_values(geometry, piece.num_points), np_, point_index)
        grown[g] = (piece, topology, spread, implicit, p_owner, c_owner, own_points, own_cells)

    for kind in ("points", "cells"):
        plans[kind].run({g: grown[g][2][kind] for g in blocks.local})

    out = []
    for g in blocks.local:
        piece, topology, spread, implicit, p_owner, c_owner, own_points, own_cells = grown[g]
        with blocks.device(g):
            if implicit:
                geometry = _grown_geometry(piece, rectilinear, grown_lower[g], topology,
                                           lower[g], blocks.device(g))
            else:
                geometry = spread["points"]["\0positions"]
            point_marks = spread["points"]["\0marks"].numpy() | np.where(
                p_owner != g, Ghosts.DUPLICATEPOINT, 0).astype(np.uint8)
            cell_marks = spread["cells"]["\0marks"].numpy() | np.where(
                c_owner != g, Ghosts.DUPLICATECELL, 0).astype(np.uint8)
            data = DataSet(topology, geometry, dtype=piece.dtype,
                           ghosts=Ghosts(cells=cell_marks, points=point_marks),
                           **piece._frame())
            point_index = tack.array(own_points, tack.i64)
            if kind_of(piece.geometry) == "nodes":
                data.geometry = _grown_nodes(data, piece.geometry, GEOMETRY, own_cells,
                                             point_index, spread)
            for name in names:
                field = piece.fields[name]
                space = field.space
                if kind_of(field) in ("nodes", "blocks"):
                    data.fields[name] = _grown_nodes(data, field, field_label(name), own_cells,
                                                     point_index, spread)
                elif kind_of(field) == "points":
                    data.fields[name] = Field(H1(data) if isinstance(space, H1)
                                              else Values(data, "points"),
                                              spread["points"][field_label(name)], field.unit)
                else:
                    data.fields[name] = Field(Constant(data) if isinstance(space, Constant)
                                              else Values(data, "cells"),
                                              spread["cells"][field_label(name)],
                                              field.unit)
            if "\0ids" in spread["points"] or "\0ids" in spread["cells"]:
                data.global_ids = GlobalIds(cells=spread["cells"].get("\0ids"),
                                            points=spread["points"].get("\0ids"))
            for name, members in piece.sets.items():
                if members.on == "points":
                    data.sets[name] = Set(data, "points", own_points[members.ids.numpy()])
                elif members.on == "cells":
                    data.sets[name] = Set(data, "cells", own_cells[members.ids.numpy()])
        data._ghosts_lost = piece._ghosts_lost
        out.append(data)
    plans.update(identities=identities, layers=layers, nodes={})
    # The ghosts of the fields of higher order (and a curved geometry) from their owners.
    higher = [n for n in names if kind_of(pieces[0].fields[n]) in ("nodes", "blocks")] \
        if pieces else []
    if pieces and kind_of(pieces[0].geometry) == "nodes":
        higher.append(GEOMETRY)
    if blocks.all_reduce({g: len(higher) for g in blocks.local}, "max"):
        refresh(blocks, out, plans, higher)
    return out, plans


def _grown_nodes(data, field, label, own_cells, point_index, spread):
    """A field of higher order on the grown block ``data``: its space remade there and
    the piece's own cells' nodes placed (an ``H1`` field's own points' values too);
    the ghosts' values are left for their owners' (``_node_plans``)."""
    space = field.space
    cells = np.arange(space.topology.num_cells, dtype=np.int64)
    own = {"cell_at": tack.array(own_cells, tack.i64)}
    add_nodes(own, label, field, cells, own["cell_at"].device)
    if isinstance(space, H1):
        made = H1(data, **space_spec(space, None))
        points = arrays.materialize(_point_values(field.values, space.topology.num_points))
        out = tack.empty((data.num_points,), points.dtype)
        scatter(points, point_index, out)
        return Field(made, nodes_placed(made, field.values, [own], label, out), field.unit)
    spec = space_spec(space, lambda values: None)
    if space.varies:
        spec["order"] = spread["cells"]["\0order " + label]
    made = L2(data, **spec)
    return Field(made, nodes_placed(made, field.values, [own], label), field.unit)


def _point_values(values, n):
    from tack.dataset.carry import _point_values as values_at_points

    return values_at_points(values, n)


def _overlap(lower, upper):
    for a in range(len(lower)):
        for b in range(a + 1, len(lower)):
            if np.all(np.maximum(lower[a], lower[b]) < np.minimum(upper[a], upper[b])):
                return True
    return False


def _axes(blocks, pieces, lower):
    """For rectilinear pieces, each topology axis's coordinates over the whole box,
    ``{axis: {global index: coordinate}}``, from every piece's own axes; ``None``
    when the pieces are not rectilinear."""
    def mine(piece, offset):
        coordinates = piece.geometry.values
        if not isinstance(coordinates, CartesianProduct):
            return None
        axes = [a for a, d in enumerate(coordinates.dims) if d != 1]
        lines = (coordinates.x, coordinates.y, coordinates.z)
        return {i: (axis, int(offset[i]), lines[axis].numpy().astype(np.float64))
                for i, axis in enumerate(axes)}
    every = blocks.all_gather({g: mine(p, p.topology.offset)
                               for g, p in zip(blocks.local, pieces)})
    if any(e is None for e in every):
        return None
    out = {}
    for part in every:
        for i, (axis, start, values) in part.items():
            line = out.setdefault(i, (axis, {}))[1]
            for k, v in enumerate(values.tolist()):
                line.setdefault(start + k, v)
    return out


def _grown_geometry(piece, rectilinear, grown_lower, topology, lower, device):
    """The implicit coordinates of a piece grown to ``topology``'s block."""
    coordinates = piece.geometry.values
    point_dims = topology.point_dims
    if isinstance(coordinates, UniformCoordinates):
        axes = [a for a, d in enumerate(coordinates.dims) if d != 1]
        return _block_geometry(coordinates, axes, list(grown_lower - lower), point_dims,
                               device)
    lines = [np.zeros(1), np.zeros(1), np.zeros(1)]
    for i, d in enumerate(point_dims):
        axis, line = rectilinear[i]
        lines[axis] = np.array([line[int(grown_lower[i]) + k] for k in range(d)])
    for axis, values in enumerate((coordinates.x, coordinates.y, coordinates.z)):
        if math.prod(lines[axis].shape) == 1 and values.shape[0] == 1:
            lines[axis] = values.numpy().astype(np.float64)
    return CartesianProduct(*lines, dtype=arrays.dtype_of(coordinates), device=device)
