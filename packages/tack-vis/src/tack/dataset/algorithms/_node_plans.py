# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Plans that give a field of higher order's ghost values from their owners: every
value of an ``H1`` field above order 1 or of an ``L2`` field that a grown piece
holds but does not own, from the piece that does. ``ghost_cells`` fills a
structured piece's ghosts with them and ``update_ghosts`` refreshes any piece's.

A value's owner is the lowest piece holding it, as a point's is: the lowest
piece with a cell of its own that has the node, so a node on an edge or face
pieces share is owned with that edge's or face's points, and the field stays
one value along it. Copies of a node are found by a key every piece computes
alike from its own cells, whichever way round they list the node's entity:

- a node inside an edge or face, by the identities of the entity's corners
  (``(owner, index there)``, as ``ghost_cells`` keeps them) and its integer
  weights on them (``(p - a, a)`` along an edge, a triangle's barycentric and
  a quad's bilinear steps), corners sorted by identity;
- a node inside a cell, and every value of an ``L2`` cell, by the cell's
  identity and its place among the cell's values (with their count);
- a corner, as its point: its identity names the owner and its index there.

The keys are sorted over every block (``_matching.match``) and each copy hears
the others: ``layers`` steps of each piece's own cells from what its ghosts
touch hold every node any other piece's ghosts have."""

import numpy as np

import tack
from tack import arrays
from tack.dataset import _matching, bases
from tack.dataset.algorithms._cell_nodes import _ramp, value_places
from tack.dataset.dataset import field_label
from tack.dataset.spaces import H1, L2
from tack.dataset.topology import StructuredTopology, UnstructuredTopology

__all__ = ["GEOMETRY", "node_plan", "refresh", "spec_key"]

#: The name ``refresh`` gives a piece's geometry.
GEOMETRY = "\0geometry"

_EDGE, _FACE, _INSIDE = 1, 2, 3
_WIDTH = tack.constant(8)       # the most corners a reference cell has


@tack.kernel
def _corner_rows(cells, code, rows, codes):
    for c in cells:
        e = cells.entity_id(c)
        codes[e] = code
        for j in range(cells.NUM_POINTS):
            rows[e * _WIDTH + j] = cells.point_id(c, j)


def _corners(topology):
    """Each cell's corners' point ids, ``(num_cells, 8)`` (-1 past its corners), each
    cell's group in ``shapes``, and the groups' shapes: host arrays."""
    n = topology.num_cells
    rows = tack.full((max(n, 1) * _WIDTH,), -1, tack.i64)
    codes = tack.zeros((max(n, 1),), tack.i32)
    shapes = []
    for group in topology.groups():
        if group.count:
            _corner_rows(group.view(), len(shapes), rows, codes)
            shapes.append(group.shape)
    return (rows.numpy()[:n * _WIDTH].reshape(n, _WIDTH).astype(np.int64),
            codes.numpy()[:n].astype(np.int64), shapes)


def _cell_points(topology):
    """Each cell's points, ``(offsets, connectivity)`` on the host (a polyhedron's
    distinct points)."""
    if not isinstance(topology, (UnstructuredTopology, StructuredTopology)):
        offsets, connectivity = topology.cell_points()
    elif isinstance(topology, UnstructuredTopology):
        offsets, connectivity = topology.offsets, topology.connectivity
    else:
        rows, _, _ = _corners(topology)
        sizes = (rows >= 0).sum(axis=1)
        return np.concatenate(([0], np.cumsum(sizes))), rows[rows >= 0]
    offsets = np.asarray(arrays.to_host(offsets), np.int64).reshape(-1)[:topology.num_cells + 1]
    return offsets, np.asarray(arrays.to_host(connectivity), np.int64).reshape(-1)


def _band(topology, own, layers):
    """The cells whose nodes a piece offers or asks about: every ghost cell, and its
    own cells within ``layers`` steps of the points the ghosts use."""
    offsets, connectivity = _cell_points(topology)
    cell_of = np.repeat(np.arange(len(own)), np.diff(offsets))
    reached = np.zeros(topology.num_points, bool)
    reached[connectivity[~own[cell_of]]] = True
    chosen = ~own
    for _ in range(layers):
        touching = np.zeros(len(own), bool)
        touching[cell_of[reached[connectivity]]] = True
        chosen = chosen | (touching & own)
        reached[connectivity[chosen[cell_of]]] = True
    return np.flatnonzero(chosen)


def _keyed(kind, ids, weights):
    """Key columns of rows of ``(identity, weight)`` pairs, sorted by identity and
    padded to four: ``kind``, four identities, four weights (u64)."""
    order = np.argsort(ids, axis=1, kind="stable")
    ids = np.take_along_axis(ids, order, axis=1)
    weights = np.take_along_axis(weights, order, axis=1)
    m, k = ids.shape
    pad = np.zeros((m, 4 - k), np.int64)
    ids, weights = np.hstack([ids, pad]), np.hstack([weights, pad])
    return ([np.full(m, kind, np.uint64)] + [ids[:, j].view(np.uint64) for j in range(4)]
            + [weights[:, j].astype(np.uint64) for j in range(4)])


def _inside(cell_identity, place, count):
    """Key columns of values inside cells: the cell's identity, the value's place and
    the cell's count of values."""
    m = len(place)
    zero = np.zeros(m, np.uint64)
    return [np.full(m, _INSIDE, np.uint64), np.asarray(cell_identity, np.int64).view(np.uint64),
            np.asarray(place, np.uint64), zero, zero, np.asarray(count, np.uint64), zero, zero,
            zero]


def _node_keys(space, values, point_identity, cell_identity, cells):
    """The values of ``cells`` other than their corners: ``(index, columns, cell)``,
    a row for each cell's node (a value several cells share has a row for each)."""
    sizes, starts, places = value_places(space, values)
    if isinstance(space, L2):
        counts = sizes[cells]
        cell = np.repeat(cells, counts)
        place = _ramp(counts)
        index = places[np.repeat(starts[cells], counts) + place]
        return index, _inside(cell_identity[cell], place, sizes[cell]), cell
    corners, codes, shapes = _corners(space.topology)
    p = space.order
    index, columns, cell = [], [], []
    for code, shape in enumerate(shapes):
        mine = cells[codes[cells] == code]
        if not len(mine):
            continue
        stored = space.stored_nodes(shape)
        kind, entity, a, b = (np.asarray(t)[stored] for t in bases.dof_tables(shape, p)[:4])
        if not np.array_equal(sizes[mine], np.full(len(mine), len(stored))):
            raise ValueError(f"{space!r} lists {sizes[mine][0]} nodes on a "
                             f"{shape.__name__} cell, not {len(stored)}")
        _, edges, faces = bases._entities(shape)
        ids = point_identity[corners[mine]]
        at = places[starts[mine][:, None] + np.arange(len(stored))]
        for j in range(len(stored)):
            if kind[j] == bases.CORNER:
                continue
            if kind[j] == bases.EDGE:
                rows = list(edges[entity[j]])
                w = [p - a[j], a[j]]
                keys = _keyed(_EDGE, ids[:, rows], np.tile(w, (len(mine), 1)))
            elif kind[j] == bases.FACE:
                rows = list(faces[entity[j]])
                aj, bj = int(a[j]), int(b[j])
                w = ([p - aj - bj, aj, bj] if len(rows) == 3 else
                     [(p - aj) * (p - bj), aj * (p - bj), aj * bj, (p - aj) * bj])
                keys = _keyed(_FACE, ids[:, rows], np.tile(w, (len(mine), 1)))
            else:
                keys = _inside(cell_identity[mine], np.full(len(mine), a[j]),
                               np.full(len(mine), len(stored)))
            index.append(at[:, j])
            columns.append(keys)
            cell.append(mine)
    if not index:
        return np.zeros(0, np.int64), [np.zeros(0, np.uint64)] * 9, np.zeros(0, np.int64)
    return (np.concatenate(index), [np.concatenate(c) for c in zip(*columns)],
            np.concatenate(cell))


def _entries(g, field, identity, layers):
    """Piece ``g``'s copies of its band's values for the matching: ``(index, columns,
    held)``, one row per value, ``held`` where a cell of its own has it."""
    space = field.space
    point_identity, cell_identity, own = identity
    cells = _band(space.topology, own, layers)
    index, columns, cell = _node_keys(space, arrays.materialize(field.values),
                                      point_identity, cell_identity, cells)
    held = own[cell]
    order = np.lexsort((~held, index))          # by value, a holding row first
    index, held = index[order], held[order]
    columns = [c[order] for c in columns]
    first = np.r_[True, index[1:] != index[:-1]] if len(index) else np.zeros(0, bool)
    lead = np.maximum.accumulate(np.where(first, np.arange(len(index)), 0))
    for c in columns:
        if not np.array_equal(c, c[lead]):
            raise AssertionError(f"piece {g}'s cells name one value of {space!r} two ways")
    return index[first], [c[first] for c in columns], held[first].astype(np.uint8)


def node_plan(blocks, fields, identities, layers):
    """The plan giving each grown piece's values of ``fields[g]`` (``{gid: Field}``,
    an ``H1`` field above order 1 or an ``L2`` field, the same space on every
    piece) that it does not own from their owners. ``identities[g]`` is ``(point
    identity, cell identity, own cells)``: each point's and cell's ``(owner,
    index there)`` packed as ``_matching`` packs them, and which cells are the
    piece's own."""
    entries = {g: _entries(g, fields[g], identities[g], layers) for g in blocks.local}
    found = _matching.match(blocks, entries)
    receives, unowned = {}, {}
    for g in blocks.local:
        index, _, held = entries[g]
        wanted = {}
        space = fields[g].space
        if isinstance(space, H1):
            # The corners: each point from its owner, a value's index its point's.
            point_identity = identities[g][0]
            wanted = _matching.from_owners(g, point_identity, np.arange(len(point_identity)))
        rows, other, other_index, other_held = found[g]
        keep = other_held == 1
        rows, other, other_index = rows[keep], other[keep], other_index[keep]
        order = np.lexsort((other, rows))       # the lowest holder first
        rows, other, other_index = rows[order], other[order], other_index[order]
        first = np.r_[True, rows[1:] != rows[:-1]] if len(rows) else np.zeros(0, bool)
        rows, other, other_index = rows[first], other[first], other_index[first]
        at = np.searchsorted(index, rows)
        owner = np.where(held == 1, g, -1).astype(np.int64)
        lower = (held[at] == 0) | (other < g)     # a lower holder, or the only one
        owner[at[lower]] = other[lower]
        theirs = np.full(len(index), -1, np.int64)
        theirs[at[lower]] = other_index[lower]
        unowned[g] = int((owner < 0).sum())
        for source in np.unique(owner[(owner != g) & (owner >= 0)]).tolist():
            pick = owner == source
            before = wanted.get(int(source), (np.zeros(0, np.int64), np.zeros(0, np.int64)))
            wanted[int(source)] = (np.concatenate([before[0], theirs[pick]]),
                                   np.concatenate([before[1], index[pick]]))
        receives[g] = wanted
    # Every rank's say, so that all of them raise rather than one leaving the rest
    # in the exchange.
    lost = blocks.all_reduce(unowned) if blocks.count else 0
    if lost:
        raise ValueError(f"the pieces' ghosts hold {int(lost)} values of a field of higher "
                         "order that no piece owns: the pieces' fields differ in their spaces (an order "
                         "per cell that differs between a cell and its ghost)")
    return _matching.plan(blocks, receives)


def spec_key(space):
    """What numbers a space's values on a given mesh, as a hashable value; ``None``
    for a space a per-cell array numbers (an order per cell, a DOF table), whose
    plan is made afresh."""
    if isinstance(space, H1):
        return None if space.dofs is not None else ("H1", space.order, space.basis)
    if space.varies:
        return None
    return ("L2", space.order, space.basis, space.layout)


def _field(piece, name):
    return piece.geometry if name == GEOMETRY else piece.fields[name]


def refresh(blocks, pieces, plans, names):
    """The fields of higher order ``names`` (this rank's; ``GEOMETRY`` the pieces'
    geometry) refreshed from their owners through a plan per space, kept in
    ``plans["nodes"]`` (but for a space numbered by a per-cell array): on every rank
    alike, by the spaces every piece reports, so a rank without pieces joins the
    same exchanges."""
    def key(name, space):
        made = spec_key(space)
        return made if made is not None else ("each", name, type(space).__name__)

    reported = blocks.all_gather({g: [(n, key(n, _field(p, n).space)) for n in names]
                                  for g, p in zip(blocks.local, pieces)})
    if any(r != reported[0] for r in reported):
        raise ValueError("the ghosts of a field of higher order are refreshed in one space "
                         "on every piece; the pieces' spaces differ")
    groups = {}
    for name, made in (reported[0] if reported else []):
        groups.setdefault(made, []).append(name)
    for made, members in groups.items():
        kept = made[0] != "each"
        plan = plans["nodes"].get(made) if kept else None
        if plan is None:
            plan = node_plan(blocks, {g: _field(p, members[0])
                                      for g, p in zip(blocks.local, pieces)},
                             plans["identities"], plans["layers"])
            if kept:
                plans["nodes"][made] = plan
        plan.run({g: {(n if n == GEOMETRY else field_label(n)): _field(p, n).values
                      for n in members} for g, p in zip(blocks.local, pieces)})
