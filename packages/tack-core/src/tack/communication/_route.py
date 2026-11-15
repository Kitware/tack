# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Routing points to the blocks that hold them, and answers back."""

import numpy as np

import tack
from tack.lang.array import Array

from ._kernels import gather, scatter


class Routing:
    """Where each point goes: the lowest block that holds it.

    ``positions`` is ``{gid: points}`` for every block on this rank, an
    ``(n, d)`` host array of coordinates each (``n`` may be 0); ``boxes``
    ``{gid: (lower, upper)}`` for every block on this rank, ``None`` for a block
    that holds nothing. A point is asked of every block whose box holds it (with
    a relative slack of 1e-6), and ``holds(gid, points)`` -- called by the block
    asked, with the points it was asked about as an ``(m, d)`` float64 host
    array -- says how well it holds each: 0 not at all, or a rank (``True`` is
    1). The point goes to the block of the highest rank, the lowest such block
    on a tie: a locator that holds a point beats one only within its tolerance. Coordinates travel as their f64 bits in ``u64`` arrays, which
    every device stores, so a block decides on exactly the points it was given.

    ``destination`` is ``{gid: array}``, each point's block, or -1 where no block
    holds it. ``send`` moves values with the points to their destinations, and
    ``reply`` returns values from there to where the points came from.
    """

    def __init__(self, blocks, positions, boxes, holds):
        self.blocks = blocks
        blocks._every_local(positions, "Routing's positions")
        blocks._every_local(boxes, "Routing's boxes")
        self._positions = {g: _points(positions[g]) for g in blocks.local}
        every = blocks.all_gather({g: None if boxes[g] is None else
                                   tuple(np.asarray(c, np.float64) for c in boxes[g])
                                   for g in blocks.local})

        # 1. Each point to every block whose box holds it.
        asked = {}
        for g in blocks.local:
            x = self._positions[g]
            asked[g] = {}
            for d, rows in _in_boxes(x, every).items():
                if len(rows):
                    message = _bits(x[rows], blocks.device(g))
                    message["slot"] = tack.array(rows.astype(np.int64), tack.i64,
                                                 device=blocks.device(g))
                    asked[g][d] = message
        questions = blocks.exchange(asked)

        # 2. Each block says which it holds.
        answers = {}
        for g in blocks.local:
            answers[g] = {}
            for origin, message in questions[g].items():
                held = np.asarray(holds(g, _unbits(message))).astype(np.int32).reshape(-1)
                answers[g][origin] = {"slot": message["slot"],
                                      "held": tack.array(held, tack.i32,
                                                         device=blocks.device(g))}
        replies = blocks.exchange(answers)

        # 3. The highest rank takes it, the lowest block on a tie (replies come
        # in order of block, so a later one must rank strictly higher).
        self.destination = {}
        for g in blocks.local:
            n = len(self._positions[g])
            destination = np.full(n, -1, np.int64)
            best = np.zeros(n, np.int32)
            for d, reply in replies[g].items():
                slots = reply["slot"].numpy()
                rank = reply["held"].numpy()
                better = rank > best[slots]
                destination[slots[better]] = d
                best[slots[better]] = rank[better]
            self.destination[g] = destination
        self._arrivals = None

    def send(self, values=None):
        """Move ``values`` (``{gid: {name: array}}``, an element per point, on the
        block's device; may be omitted) with the points to their destinations.
        Returns ``{gid: (positions, {name: array})}`` for every block on this
        rank: what arrived there, the points as an ``(m, d)`` host array and the
        values on the block's device, in order of origin block and then of the
        points there. ``reply`` answers in that order."""
        blocks = self.blocks
        values = values or {g: {} for g in blocks.local}
        outgoing, self._order = {}, {}
        for g in blocks.local:
            outgoing[g] = {}
            for d, rows in _by_destination(self.destination[g]):
                device = blocks.device(g)
                message = _bits(self._positions[g][rows], device)
                message["slot"] = tack.array(rows.astype(np.int64), tack.i64, device=device)
                indices = tack.array(rows.astype(np.int64), tack.i64, device=device)
                for name, array in values.get(g, {}).items():
                    if not isinstance(array, Array):
                        raise TypeError(f"Routing.send() moves Tack arrays; block {g}'s "
                                        f"{name!r} is a {type(array).__name__}")
                    message["value." + name] = gather(array, indices)
                outgoing[g][int(d)] = message
        incoming = blocks.exchange(outgoing)
        arrived = {}
        self._arrivals = {}
        for g in blocks.local:
            parts = list(incoming[g].items())               # in order of origin
            self._arrivals[g] = [(origin, m["slot"], m["slot"].size) for origin, m in parts]
            x = (np.concatenate([_unbits(m) for _, m in parts]) if parts
                 else np.zeros((0, self._width()), np.float64))
            names = sorted({k for _, m in parts for k in m if k.startswith("value.")})
            joined = {}
            for name in names:
                pieces = [m[name] for _, m in parts]
                joined[name[len("value."):]] = pieces[0] if len(pieces) == 1 \
                    else tack.concat(pieces)
            arrived[g] = (x, joined)
        return arrived

    def reply(self, values, like):
        """Send ``values`` -- ``{gid: {name: array}}``, an element for each point
        that arrived at the block (``send``'s order) -- back to where the points
        came from. Returns ``{gid: {name: array}}`` for the blocks on this rank,
        an element per point of the block's ``positions``, zero where the point
        had no destination. ``like`` is ``{name: element type}``, for blocks
        none of whose points were taken."""
        if self._arrivals is None:
            raise ValueError("Routing.reply() answers what Routing.send() delivered: "
                             "send first")
        blocks = self.blocks
        outgoing = {}
        for g in blocks.local:
            outgoing[g] = {}
            start = 0
            for origin, slots, count in self._arrivals[g]:
                indices = tack.array(np.arange(start, start + count, dtype=np.int64),
                                     tack.i64, device=blocks.device(g))
                message = {"slot": slots.to_device(blocks.device(g))}
                for name in like:
                    message["value." + name] = gather(values[g][name], indices)
                outgoing[g][origin] = message
                start += count
        incoming = blocks.exchange(outgoing)
        out = {}
        for g in blocks.local:
            n = len(self._positions[g])
            out[g] = {}
            for name, dtype in like.items():
                array = tack.zeros((n,), dtype, device=blocks.device(g))
                for _, message in incoming[g].items():
                    scatter(message["value." + name], message["slot"], array)
                out[g][name] = array
        return out

    def _width(self):
        widths = [x.shape[1] for x in self._positions.values() if x.ndim == 2]
        return widths[0] if widths else 3


def _by_destination(destination):
    """``(d, rows)`` for each destination ``d`` of ``destination`` (-1 for none left
    out), in order of ``d``, each one's rows ascending: one stable sort, not a
    pass over every point for each destination."""
    taken = np.flatnonzero(destination >= 0)
    if not len(taken):
        return []
    keys = destination[taken]
    small = np.int16 if keys.max() < 2 ** 15 else np.int64     # sorted by radix
    order = taken[np.argsort(keys.astype(small), kind="stable")]
    sorted_keys = destination[order]
    cuts = np.flatnonzero(np.diff(sorted_keys)) + 1
    return [(int(sorted_keys[run[0]]), order[run[0]:run[1]])
            for run in zip(np.concatenate(([0], cuts)), np.concatenate((cuts, [len(order)])))]


def _widened(box):
    """A box ``(lower, upper)`` widened by its slack: 1e-6 of its largest side, at
    least 1e-6."""
    lower, upper = box
    slack = 1e-6 * max(float(np.max(upper - lower)) if len(upper) else 0.0, 1.0)
    return lower - slack, upper + slack


def _inside(columns, rows, lower, upper):
    """Which of ``rows`` (indices into ``columns``, the points' coordinates by axis)
    lie in ``[lower, upper]``, every axis."""
    keep = np.ones(len(rows), bool)
    for axis in range(len(lower)):
        values = columns[axis][rows]
        keep &= (values >= lower[axis]) & (values <= upper[axis])
    return rows[keep]


#: Boxes up to this many are tested against every point; more are binned.
_DIRECT_BOXES = 4
#: Grid cells a box, about: a box's points and those of the cells its corners
#: cut, which it also tests, about (1 + 1 / 16 ** (1 / 3)) ** 3 = 2 of its own
#: (8 with 2 cells a box).
_CELLS_PER_BOX = 16


def _in_boxes(x, boxes):
    """``{d: rows}``: the points of ``x`` (an ``(n, d)`` float64 array) in each box of
    ``boxes`` (``(lower, upper)``, or ``None`` for a block holding nothing),
    widened by its slack (``_widened``), the rows in no particular order (a
    point's rank is its own); boxes holding none of them left out.

    Few boxes are tested against every point. More are binned on a grid over
    their union, about ``_CELLS_PER_BOX`` cells a box (at most 2**16 cells, so
    the points sort by cell as 16-bit keys): each point is tested only against
    the boxes of its cell. A point and a box's corners are placed by the same arithmetic,
    which rounding keeps in order, so no point in a box falls outside its
    cells. Testing every point against every box was 2.6 s of a probe of 2M
    points over 64 pieces."""
    n = len(x)
    widened = {d: _widened(box) for d, box in enumerate(boxes) if box is not None}
    if not n or not widened:
        return {}
    columns = np.ascontiguousarray(x.T)
    every = np.arange(n, dtype=np.int64)
    if len(widened) <= _DIRECT_BOXES:
        found = {d: _inside(columns, every, *box) for d, box in widened.items()}
        return {d: rows for d, rows in found.items() if len(rows)}
    lows = np.array([box[0] for box in widened.values()])
    highs = np.array([box[1] for box in widened.values()])
    origin, extent = lows.min(axis=0), highs.max(axis=0) - lows.min(axis=0)
    dims_ = x.shape[1]
    live = extent > 0
    side = (np.prod(extent[live]) / (_CELLS_PER_BOX * len(widened))) ** (
        1.0 / max(int(live.sum()), 1)) if live.any() else 1.0
    dims = np.where(live, np.clip(np.ceil(extent / max(side, 1e-300)), 1, 2 ** 16), 1)
    while np.prod(dims) > 2 ** 16 - 1:
        dims = np.maximum(np.ceil(dims / 2), 1)
    dims = dims.astype(np.int64)
    inverse = np.where(live, dims / np.where(live, extent, 1.0), 0.0)

    def cells(values, axis):
        with np.errstate(invalid="ignore"):            # NaNs: outside, keyed apart
            return np.clip(np.floor((values - origin[axis]) * inverse[axis]), 0,
                           dims[axis] - 1).astype(np.int64)

    # Each point's cell (outside the union: in no box), then the points by cell.
    within = np.ones(n, bool)
    key = np.zeros(n, np.int64)
    stride = 1
    for axis in range(dims_):
        within &= (columns[axis] >= origin[axis]) & (columns[axis] <= origin[axis]
                                                     + extent[axis])
        key += cells(columns[axis], axis) * stride
        stride *= int(dims[axis])
    key = np.where(within, key, stride).astype(np.uint16 if stride < 2 ** 16 else np.int64)
    order = np.argsort(key, kind="stable")
    starts = np.searchsorted(key[order], np.arange(stride + 1))
    found = {}
    for d, (lower, upper) in widened.items():
        first = [int(cells(np.array([lower[a]]), a)[0]) for a in range(dims_)]
        last = [int(cells(np.array([upper[a]]), a)[0]) for a in range(dims_)]
        ranges = [np.arange(first[a], last[a] + 1) for a in range(dims_)]
        grid = np.zeros(1, np.int64)
        stride = 1
        for a in range(dims_):
            grid = (grid[:, None] + ranges[a][None, :] * stride).reshape(-1)
            stride *= int(dims[a])
        if not len(grid):
            continue
        candidates = np.concatenate([order[starts[c]:starts[c + 1]] for c in grid])
        rows = _inside(columns, candidates, lower, upper)
        if len(rows):
            found[d] = rows
    return found


def _points(points):
    """Points as an ``(n, d)`` float64 host array."""
    x = np.asarray(points, np.float64)
    return x if x.ndim == 2 else x.reshape(len(x), -1)


def _bits(x, device):
    bits = np.ascontiguousarray(np.asarray(x, np.float64)).view(np.uint64)
    bits = bits.reshape(len(x), -1)
    return {f"x{axis}": tack.array(bits[:, axis].copy(), tack.u64, device=device)
            for axis in range(bits.shape[1])}


def _unbits(message):
    axes = sorted(int(k[1:]) for k in message if k.startswith("x") and k[1:].isdigit())
    if not axes:
        return np.zeros((0, 0))
    return np.stack([message[f"x{axis}"].numpy().view(np.float64) for axis in axes], axis=1)
