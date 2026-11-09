# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Partitioned datasets: a dataset in pieces, as VTK's ``vtkPartitionedDataSet``.

A ``PartitionedDataSet`` is an ordered list of ``DataSet``s. Each piece is a
whole dataset -- its own topology, numbering, geometry, fields, sets and
device -- so work that needs one piece at a time takes the piece: the caller
iterates over them. Only algorithms whose answer depends on more than one
piece take the partitioned dataset itself: ``statistics`` and ``entropy``
combine what each piece holds, and ``connected_components``, ``probe``,
``advect``, ``outliers``, ``voxel_downsample``, ``deposit`` and
``halo_catalog`` combine what pieces hold near each other. ``split`` makes one from a structured grid,
and ``tack.interop.vtk`` converts VTK's partitioned datasets and collections.

Every cell and point that several pieces hold has one owner; the other
copies are ghosts, marked in each piece's ``ghosts``
(``tack.dataset.ghosts``), and reductions count only what a piece owns.
"""

from tack.dataset.dataset import DataSet

__all__ = ["PartitionedDataSet"]


class PartitionedDataSet:
    """A dataset in pieces: an ordered list of ``DataSet``s (``partitions``), each
    with its own topology, fields, sets and device, as VTK's
    ``vtkPartitionedDataSet``. A piece may have no cells. Pieces may be on
    different devices; an algorithm given the partitioned dataset runs on each
    piece where the piece is.

    It is a container, not a dataset: iterate over it (``for piece in pds``),
    index it (``pds[i]``) or take its length.

    Across MPI ranks, it holds this rank's pieces, and ``world`` (a
    ``tack.communication.World``; this process alone by default) is where the
    others are, for the algorithms that combine pieces."""

    def __init__(self, partitions=(), world=None):
        partitions = tuple(partitions)
        for i, piece in enumerate(partitions):
            if not isinstance(piece, DataSet):
                raise TypeError(f"partition {i} is a {type(piece).__name__}, not a "
                                "tack.dataset.DataSet")
        self._partitions = partitions
        if world is None:
            from tack.communication import World
            world = World()
        self.world = world

    @property
    def partitions(self):
        """The pieces, a tuple of ``DataSet``s."""
        return self._partitions

    def __len__(self):
        return len(self._partitions)

    def __iter__(self):
        return iter(self._partitions)

    def __getitem__(self, index):
        return self._partitions[index]

    def __repr__(self):
        cells = sum(piece.num_cells for piece in self._partitions)
        return (f"PartitionedDataSet({len(self._partitions)} partitions, "
                f"{cells} cells)")


def fields_of(data, field, what):
    """``field`` in each piece of ``data``, for an algorithm that combines them
    (``what``): a list of each piece's ``Field``, resolved once a piece. ``field`` is
    a name, or an object whose ``on(piece)`` gives the piece's ``Field``; a
    ``Field`` belongs to one piece's topology and is refused, as is a piece
    without the name."""
    from tack.dataset.algorithms._common import _field
    from tack.dataset.dataset import is_field_key

    named = is_field_key(field)
    if not named and not callable(getattr(field, "on", None)):
        raise TypeError(f"{what} of a PartitionedDataSet takes a field's name (or name "
                        "and space) or an object whose on(piece) gives the piece's: a "
                        "Field belongs to one partition's topology")
    out = []
    for i, piece in enumerate(data):
        if named and field not in piece.fields:
            raise KeyError(f"partition {i} has no field {field!r}: it has "
                           f"{', '.join(map(repr, piece.fields)) or 'none'}")
        out.append(_field(piece, field))
    return out


def agree(world, check, *args, **kwargs):
    """``check(*args, **kwargs)``, run on this rank, with its outcome agreed on
    by every rank of ``world``: every rank's result, in order of rank, if it
    raised on none, else an error on every rank. The rank that raised raises
    its own exception; the others raise one of the same kind and message, with
    a note naming the rank. Collective: one gather of the results, which
    should be small, so every rank must call it.

    An algorithm over pieces checks its arguments against the pieces it holds
    (a field's kind, width and space, the pieces' cell kinds). A rank holding
    no pieces, or pieces that pass, cannot know another's failed, and would go
    on into the next collective and wait there forever. Checks run through
    ``agree`` before the algorithm's first collective fail on every rank
    instead. ``check`` must make no collective call itself."""
    result, error, said = _attempt(check, *args, **kwargs)
    gathered = world._allgather((result, said))
    _agreed([s for _, s in gathered], error)
    return [r for r, _ in gathered]


def _attempt(check, *args, **kwargs):
    """``(result, error, said)``: what ``check`` returned or raised, and what
    this rank tells the others of it (``None``, or the error's module, class
    name and message)."""
    try:
        return check(*args, **kwargs), None, None
    except Exception as error:
        kind = type(error)
        message = (error.args[0] if len(error.args) == 1 and isinstance(error.args[0], str)
                   else str(error))
        return None, error, (kind.__module__, kind.__qualname__, message)


def _agreed(said, error):
    """Raise if any rank's check raised (``said``, gathered from ``_attempt``):
    this rank's own ``error`` if it has one, else the first rank's, remade."""
    if error is not None:
        raise error
    rank = next((r for r, s in enumerate(said) if s is not None), None)
    if rank is not None:
        raise _remade(*said[rank], rank, len(said))


def _remade(module, name, message, rank, size):
    """The exception another rank raised, of its class where this rank can import
    it and make one from a message (else a ``RuntimeError``)."""
    import importlib

    kind = RuntimeError
    try:
        found = importlib.import_module(module)
        for part in name.split("."):
            found = getattr(found, part)
        if isinstance(found, type) and issubclass(found, Exception):
            kind = found
    except Exception:
        pass
    try:
        error = kind(message)
    except Exception:
        error = RuntimeError(message)
    error.add_note(f"Raised on rank {rank} of {size}, whose pieces failed the check; "
                   "every rank raises it.")
    return error


def common_origin(data, what, world=None, check=None):
    """The origin every piece of ``data`` has, on every rank of ``world``
    (``data.world`` by default): ``what`` (an algorithm that compares or moves
    positions between pieces) refuses pieces on two origins, in two spatial
    units, or at two times. ``(0.0, 0.0, 0.0)``
    with no pieces anywhere.

    ``check``, a function of no arguments, runs first and is agreed on in the
    same gather, as ``agree`` does: checks of the pieces whose algorithm's
    first collective is this one cost no other."""
    from tack.dataset import units

    world = data.world if world is None else world
    _, error, said = (None, None, None) if check is None else _attempt(check)
    mine = sorted({piece.origin for piece in data})
    spatial = sorted({piece.unit or "" for piece in data})
    times = sorted({(piece.time, piece.time_unit) for piece in data}, key=repr)
    gathered = world._allgather((mine, spatial, times, said))
    _agreed([part for *_, part in gathered], error)
    gathered = [g[:3] for g in gathered]
    every = sorted({o for part, _, _ in gathered for o in part})
    units.common(f"{what}'s pieces' geometry",
                 [u or None for _, part, _ in gathered for u in part])
    _one_time(f"{what}'s pieces", [t for _, _, part in gathered for t in part])
    if len(every) > 1:
        raise ValueError(f"{what} needs its pieces on one origin, not {every}: rebase "
                         "them to one first (DataSet.rebased)")
    return every[0] if every else (0.0, 0.0, 0.0)


def _one_time(what, times):
    """Refuse ``times`` (``(time, time_unit)`` pairs) of more than one step: pieces of
    one dataset are at one time, compared in one unit where both are known. A
    piece with no time is not checked."""
    import math

    from tack.dataset import units

    known = sorted({(t, u) for t, u in times if t is not None}, key=repr)
    for t, u in known[1:]:
        t0, u0 = known[0]
        if u is not None and u0 is not None and u != u0:
            factor, offset = units.conversion(u, u0)
            t = t * factor + offset
        if not math.isclose(t, t0, rel_tol=1e-12):
            raise ValueError(f"{what} are at two times, {known[0]} and {(t, u)}: pieces of "
                             "one step have one time")
