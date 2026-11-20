# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Moving fields of higher order between pieces cell by cell: each cell's values in
node order (an ``H1`` space's lexicographic nodes, as ``carry`` lists them; an ``L2``
space's in its order) sent, and placed through another space's own numbering, so a
node two cells share is one value. ``merge_pieces`` and ``ghost_cells`` move curved
geometries, ``H1`` fields above order 1 and ``L2`` fields this way."""

import numpy as np

import tack
from tack import arrays
from tack.dataset.spaces import H1, L2, Constant, Values


def field_kind(field):
    """How a field travels: ``"points"`` or ``"cells"`` (a value each), ``"nodes"``
    (an ``H1`` field above order 1: each cell's nodes, and its points' values),
    ``"blocks"`` (an ``L2`` field: each cell's values), or ``None``."""
    space = field.space
    if isinstance(space, H1) and space.order > 1:
        return "nodes"
    if isinstance(space, L2):
        return "blocks"
    if (isinstance(space, H1) and space.order == 1) or \
            (isinstance(space, Values) and space.on == "points"):
        return "points"
    if isinstance(space, Constant) or (isinstance(space, Values) and space.on == "cells"):
        return "cells"
    return None


def space_spec(space, put):
    """What remakes a field's space on the merged dataset: ``None`` for those made
    from the topology alone, an ``H1`` space's parameters (but an element-to-DOF
    table, which numbers another mesh's cells), an ``L2`` space's (an order per
    cell, an array, sent through ``put`` as ``("array", name)``)."""
    if isinstance(space, H1) and space.order > 1:
        return {k: v for k, v in space.params.items() if k != "dofs"}
    if isinstance(space, L2):
        spec = {"basis": space.basis or "gauss_lobatto",
                "layout": space.params.get("layout", "auto")}
        spec["order"] = ("array", put(space.cell_keys())) if space.varies else space.order
        return spec
    return None


@tack.kernel
def _set_sizes(cells, per_cell, out):
    for c in cells:
        out[cells.entity_id(c)] = per_cell


def _ramp(sizes):
    """``0 .. sizes[k] - 1`` for each ``k``, end to end (host)."""
    total = int(sizes.sum())
    starts = np.repeat(np.cumsum(sizes) - sizes, sizes)
    return np.arange(total, dtype=np.int64) - starts


def value_places(space, values):
    """Where each cell's nodes are in ``space``'s values, ``(sizes, starts, places)``
    on the host: cell ``c``'s node ``n`` (lexicographic, as ``carry`` lists an
    ``H1`` space's; an ``L2`` cell's in its order) at ``places[starts[c] + n]``."""
    from tack.dataset.carry import _node_entries

    topology = space.topology
    n = topology.num_cells
    if isinstance(space, L2):
        offsets = np.asarray(arrays.to_host(space.offsets), np.int64).reshape(-1)
        sizes = np.diff(offsets[:n + 1])
        if space.layout == "cells":
            # Value k of cell c at k * cells + c; listed cell by cell.
            cell = np.repeat(np.arange(n, dtype=np.int64), sizes)
            return sizes, offsets[:n], _ramp(sizes) * n + cell
        return sizes, offsets[:n], np.arange(int(offsets[n]) if n else 0, dtype=np.int64)
    index, entry, groups, _ = _node_entries(space, values)
    sizes = tack.zeros((max(n, 1),), tack.i64)
    for group, _, per_cell in groups:
        _set_sizes(group.view(), per_cell, sizes)
    return (sizes.numpy()[:n].astype(np.int64), entry.numpy()[:n].astype(np.int64),
            index.numpy().astype(np.int64))


def add_nodes(message, name, field, cells, device):
    """An owned cell's nodes of ``field``, each cell's in its order: values and
    counts, into ``message``."""
    from tack.dataset.carry import _take

    sizes, starts, places = value_places(field.space, arrays.materialize(field.values))
    cells = np.asarray(cells, np.int64)
    mine = sizes[cells]
    chosen = places[np.repeat(starts[cells], mine) + _ramp(mine)] if len(cells) else \
        np.zeros(0, np.int64)
    message[f"\0nodes {name}"] = _take(field.values,
                                       tack.array(chosen, tack.i64, device=device))
    message[f"\0sizes {name}"] = tack.array(mine, tack.i64, device=device)


def nodes_placed(space, like, messages, name, points=None):
    """The values of a field in ``space`` (on the merged mesh) from every message's
    cells' nodes, ``points`` (its values at the points, for an ``H1`` field) first."""
    from tack.communication._kernels import scatter
    from tack.dataset.carry import _copy_prefix, _like

    out = _like(like, space.size)
    if points is not None:
        _copy_prefix(points, out)
    sizes, starts, places = value_places(space, out)
    for m in messages:
        key = f"\0nodes {name}"
        if key not in m or not m["cell_at"].size:
            continue
        cells = m["cell_at"].numpy().astype(np.int64)
        came = m[f"\0sizes {name}"].numpy().astype(np.int64)
        if not np.array_equal(came, sizes[cells]):
            raise ValueError(f"merge_pieces: the cells' nodes of {name!r} do not match the "
                             "merged mesh's")
        at = places[np.repeat(starts[cells], came) + _ramp(came)]
        scatter(m[key].to_device(out.device), tack.array(at, tack.i64, device=out.device),
                out)
    return out
