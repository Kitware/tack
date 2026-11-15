# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Temporal filters: steps streamed through filters one at a time.

Tack holds no series of steps. A simulation in situ, or a reader's loop, hands
each step to the filters as a dataset with its time (``DataSet.time``, in
``DataSet.time_unit``); stateless filters treat it as any dataset, and the
temporal filters here are objects fed one step at a time (``update(data)``,
which returns that step's output), keeping what they need of earlier steps.

What they keep is a copy by default: in situ, arrays often come without a copy
(DLPack) and the simulation overwrites them at the next step. A filter copies
only what it reads -- a velocity field, positions where the mesh moves -- and
``copy=False`` keeps references where the caller knows its buffers are not
reused. Implicit arrays (an image's coordinates, a constant field) are kept as
they are: nothing writes them.

A ``PartitionedDataSet`` streams piece by piece: each piece keeps its own
state, so every step must bring the same pieces.

- ``RunningStatistics``: per point or cell, the mean, variance, minimum and
  maximum of fields over the steps so far.
- ``TimeDerivative``: each field's rate of change since the previous step.
- ``Advection``: particles carried by a velocity field through the steps, new
  ones injected at any time; ``extend_paths`` joins its particles, step after
  step, into paths; ``Pathlines`` and ``Streaklines`` are built on them.
"""

import math

import numpy as np

import tack
from tack import arrays
from tack.dataset import units
from tack.dataset.dataset import DataSet, Field, field_name, is_field_key
from tack.dataset.partitioned import PartitionedDataSet

__all__ = ["Advection", "Pathlines", "RunningStatistics", "Streaklines", "TimeDerivative",
           "extend_paths"]


def _elapsed(earlier, later, unit=None):
    """``later``'s time less ``earlier``'s, in ``unit`` (by default ``later``'s
    time unit): each a dataset or a ``_Step``. Both must have a time; their
    units are converted where both are known, and two steps of which one has a
    unit and the other none are refused."""
    for step, which in ((earlier, "the earlier step"), (later, "the later step")):
        if step.time is None:
            raise ValueError(f"{which} has no time: give each step one (DataSet.time)")
    if (earlier.time_unit is None) != (later.time_unit is None):
        raise ValueError(f"one step's time is in {earlier.time_unit or later.time_unit!r} "
                         "and the other's unknown: give both a time unit, or neither")
    t0, t1 = earlier.time, later.time
    if earlier.time_unit is not None and earlier.time_unit != later.time_unit:
        factor, offset = units.conversion(earlier.time_unit, later.time_unit)
        t0 = t0 * factor + offset
    dt = t1 - t0
    target = unit if unit is not None else later.time_unit
    if target is not None and later.time_unit is not None and target != later.time_unit:
        factor, _ = units.conversion(later.time_unit, units.time_unit(target))
        dt *= factor
    elif target is not None and later.time_unit is None:
        raise ValueError("the steps' times have no unit to convert from")
    if not math.isfinite(dt):
        raise ValueError("the time between the steps is not finite")
    return dt


def _kept(values, copy):
    """``values`` as a filter keeps them: a copy of a stored array (unless not
    ``copy``), an implicit array as it is."""
    if not copy or isinstance(values, arrays._Implicit):
        return values
    out = tack.empty_like(values)
    tack.copyto(out, values)
    return out


class _Step:
    """What a temporal filter keeps of one step: its time and time unit, its
    topology (by reference, to tell a step of another mesh), and of ``fields``
    (names) and the geometry, if ``positions``, the values -- copies, unless
    ``copy`` is false. Each kept field keeps its space and unit."""

    def __init__(self, data, fields=(), positions=False, copy=True):
        self.time = data.time
        self.time_unit = data.time_unit
        self.topology = data.topology
        self.origin = data.origin
        self.unit = data.unit
        self.fields = {}
        for name in fields:
            if name not in data.fields:
                raise KeyError(f"no field {name!r}: the step has "
                               f"{', '.join(map(repr, data.fields)) or 'none'}")
            field = data.fields[name]
            self.fields[name] = Field(field.space, _kept(field.values, copy), field.unit)
        geometry = data.geometry
        self.geometry = Field(geometry.space, _kept(geometry.values, copy),
                              geometry.unit) if positions else None

    def same_mesh(self, data):
        """Whether ``data`` is on this step's mesh: the same topology object."""
        return data.topology is self.topology


# -- the same mesh ---------------------------------------------------------------

@tack.kernel
def _differences(a, b, out):
    for i in range(out.shape[0]):
        out[i] = 1.0 if a[i] != b[i] else 0.0


def _equal(a, b):
    """Whether two id arrays (stored or implicit) hold the same values, compared on
    their device."""
    if a is b:
        return True
    if arrays.size_of(a) != arrays.size_of(b):
        return False
    n = arrays.size_of(a)
    if not n:
        return True
    device = a.device if isinstance(a, tack.Array) else b.device
    flags = tack.empty((n,), tack.f32, device=device)
    _differences(a, b, flags)
    return flags.max() == 0.0


def _same_mesh(a, b):
    """Whether topologies ``a`` and ``b`` are one mesh: the same object, or of one
    kind with the same sizes and arrays. In situ a simulation may hand each step
    a new topology over the same mesh."""
    if a is b:
        return True
    if type(a) is not type(b) or a.num_points != b.num_points or a.num_cells != b.num_cells:
        return False
    if hasattr(a, "point_dims"):
        return a.point_dims == b.point_dims and a.offset == b.offset
    return all(_equal(getattr(a, name), getattr(b, name)) for name in a._STORED)


@tack.kernel
def _moved_apart(a, b, out):
    for i in range(out.shape[0]):
        out[i] = 1.0 if any(a[i] != b[i]) else 0.0


def _same_positions(a, b):
    """Whether two geometries' values (arrays of points, stored or implicit) are
    the same points, compared on their device."""
    if a is b:
        return True
    n = arrays.size_of(a)
    if n != arrays.size_of(b) or arrays.width_of(a) != arrays.width_of(b):
        return False
    if not n:
        return True
    device = a.device if isinstance(a, tack.Array) else b.device
    flags = tack.empty((n,), tack.f32, device=device)
    _moved_apart(a, b, flags)
    return flags.max() == 0.0


def _on_one_mesh(a, b):
    """Whether datasets ``a`` and ``b`` are on one mesh at one origin, ``b`` without
    hidden or refined cells (``Ghosts.MASKED_CELLS``), which the kernel would
    search and a probe passes over."""
    from tack.dataset.ghosts import Ghosts, _has

    if b.ghosts is not None:
        with b.device:
            if _has(b.ghosts.cells, Ghosts.MASKED_CELLS):
                return False
    return (np.array_equal(np.asarray(a.origin), np.asarray(b.origin))
            and _same_mesh(a.topology, b.topology)
            and _same_positions(a.geometry.values, b.geometry.values))


def _one_mesh(before, now):
    """Whether two steps are on one mesh (``_on_one_mesh``); over pieces, each of
    ``now``'s with ``before``'s in order, agreed on every rank (collective)."""
    if not isinstance(now, PartitionedDataSet):
        return not isinstance(before, PartitionedDataSet) and _on_one_mesh(before, now)
    mine = (isinstance(before, PartitionedDataSet) and len(before) == len(now)
            and all(_on_one_mesh(a, b) for a, b in zip(before, now)))
    return all(now.world._allgather(mine))


def _refuse_another_mesh(what, kept, data):
    if not _same_mesh(kept, data.topology):
        raise ValueError(f"{what} needs one mesh over the steps, and this step is on "
                         "another: remesh it onto the first step's mesh first")


# -- streaming ------------------------------------------------------------------

class _Streamed:
    """A temporal filter over a dataset, or piece by piece over a partitioned
    dataset (each piece its own state)."""

    def __init__(self, fields):
        self.fields = [fields] if is_field_key(fields) else list(fields)
        if not self.fields:
            raise ValueError("name at least one field")
        self._states = None

    def update(self, data):
        """This step's output (``None`` where the filter has none yet)."""
        if isinstance(data, PartitionedDataSet):
            pieces = list(data)
            if self._states is None:
                self._states = [self._new_state() for _ in pieces]
            if len(pieces) != len(self._states):
                raise ValueError(f"the steps bring {len(self._states)} pieces, then "
                                 f"{len(pieces)}: each piece keeps its own state")
            outs = [self._on_device(piece, state) for piece, state in zip(pieces, self._states)]
            return None if any(o is None for o in outs) else \
                PartitionedDataSet(outs, world=data.world)
        if self._states is None:
            self._states = [self._new_state()]
        if len(self._states) != 1:
            raise ValueError("the steps brought pieces before: keep streaming pieces")
        return self._on_device(data, self._states[0])

    def _on_device(self, data, state):
        # What a step's update allocates, it allocates on the step's device.
        with data.device:
            return self._update(data, state)

    def _fields_of(self, data):
        for name in self.fields:
            if name not in data.fields:
                raise KeyError(f"no field {name!r}: the step has "
                               f"{', '.join(map(repr, data.fields)) or 'none'}")
        return [data.fields[name] for name in self.fields]


def _accumulator_type(data, values):
    """Sums run in f64 where the device has it, else in f32."""
    if tack.f64 in data.device._backend.supported_dtypes:
        return tack.f64
    return tack.f32


def _like(values, dtype):
    width = arrays.width_of(values)
    return tack.empty((arrays.size_of(values),), tack.vec(width, dtype) if width else dtype)


@tack.kernel
def _start(x, mean, m2, lo, hi):
    for i in range(mean.shape[0]):
        mean[i] = x[i]
        m2[i] = x[i] * 0.0
        lo[i] = x[i]
        hi[i] = x[i]


@tack.kernel
def _welford(x, n, mean, m2, lo, hi):
    # Welford's update, the step's value the n-th.
    for i in range(mean.shape[0]):
        v = x[i]
        delta = v - mean[i]
        mean[i] = mean[i] + delta / n
        m2[i] = m2[i] + delta * (v - mean[i])
        lo[i] = min(lo[i], v)
        hi[i] = max(hi[i], v)


@tack.kernel
def _scaled_copy(src, out, factor):
    for i in range(out.shape[0]):
        out[i] = src[i] * factor


class _Running:
    def __init__(self):
        self.topology = None
        self.count = 0
        self.sums = {}


class RunningStatistics(_Streamed):
    """Per point or cell, statistics of ``fields`` (a name or names) over the steps
    so far: each ``update(data)`` adds the step's values and returns the step with
    ``<name>_mean``, ``<name>_variance`` (the population variance, ``M2 / n``),
    ``<name>_min`` and ``<name>_max`` beside its fields, computed by Welford's
    method on the device (in f64 where the device has it). Vector fields are taken
    component by component.

    The steps must be on one mesh -- the same topology, or one of the same kind,
    sizes and arrays -- or the update is refused: remesh onto the first step's mesh
    first. Nothing of earlier steps is kept but the sums, so nothing is copied.
    The mean, minimum and maximum are in the field's unit, the variance in its
    square (kelvin squared for degrees Celsius). ``count`` is the steps added."""

    def _new_state(self):
        return _Running()

    @property
    def count(self):
        return self._states[0].count if self._states else 0

    def _update(self, data, state):
        fields = self._fields_of(data)
        if state.topology is None:
            state.topology = data.topology
        else:
            _refuse_another_mesh("RunningStatistics", state.topology, data)
        state.count += 1
        out = dict(data.fields)
        for name, field in zip(self.fields, fields):
            values = arrays.materialize(field.values)
            if arrays.dtype_of(values).numpy_dtype.kind not in "fiu":
                raise TypeError(f"field {name!r} is not numeric")
            if name not in state.sums:
                dtype = _accumulator_type(data, values)
                state.sums[name] = [_like(values, dtype) for _ in range(4)]
                if arrays.size_of(values):
                    _start(values, *state.sums[name])
            elif arrays.size_of(values):
                _welford(values, state.count, *state.sums[name])
            mean, m2, lo, hi = state.sums[name]
            n = arrays.size_of(values)
            copies = []
            for source, factor in ((mean, 1.0), (m2, 1.0 / state.count), (lo, 1.0), (hi, 1.0)):
                copy = _like(source, arrays.dtype_of(source))
                if n:
                    _scaled_copy(source, copy, factor)
                copies.append(copy)
            unit = field.unit
            square = units.derived((units.delta(unit), 2))
            for suffix, values_out, u in zip(("mean", "variance", "min", "max"), copies,
                                             (unit, square, unit, unit)):
                out[f"{field_name(name)}_{suffix}"] = Field(field.space, values_out, u)
        from tack.dataset.ghosts import keep_marks

        return keep_marks(data, DataSet(data.topology, data.geometry, fields=out,
                                        sets=dict(data.sets)))


@tack.kernel
def _rate(x, previous, inverse, out):
    for i in range(out.shape[0]):
        out[i] = (x[i] - previous[i]) * inverse


class _Previous:
    def __init__(self):
        self.step = None


class TimeDerivative(_Streamed):
    """Each of ``fields``' rate of change since the previous step: ``update(data)``
    returns the step with ``<name>_dt``, ``(value - previous) / (time - previous
    time)`` per point or cell (a backward difference, in f64 where the device has
    it), in the field's unit over the time unit (a field in degrees Celsius gives
    kelvin per second). The first step has no previous one: its update returns
    ``None``.

    The previous step's values are kept as copies (``copy=False`` keeps
    references, where the caller's buffers are not overwritten between steps).
    The steps must be on one mesh and in increasing time, with times in units
    that convert (``_elapsed``)."""

    def __init__(self, fields, copy=True):
        super().__init__(fields)
        self.copy = copy

    def _new_state(self):
        return _Previous()

    def _update(self, data, state):
        fields = self._fields_of(data)
        previous = state.step
        state.step = _Step(data, self.fields, copy=self.copy)
        if previous is None:
            return None
        _refuse_another_mesh("TimeDerivative", previous.topology, data)
        dt = _elapsed(previous, data)
        if not dt > 0:
            raise ValueError(f"the steps' times do not increase: {previous.time} then "
                             f"{data.time}")
        out = dict(data.fields)
        for name, field in zip(self.fields, fields):
            values = arrays.materialize(field.values)
            before = arrays.materialize(previous.fields[name].values)
            rate = _like(values, _accumulator_type(data, values))
            if arrays.size_of(values):
                _rate(values, before, 1.0 / dt, rate)
            unit = units.derived(units.delta(field.unit), (data.time_unit, -1))
            out[f"{field_name(name)}_dt"] = Field(field.space, rate, unit)
        from tack.dataset.ghosts import keep_marks

        return keep_marks(data, DataSet(data.topology, data.geometry, fields=out,
                                        sets=dict(data.sets)))


# -- advection ------------------------------------------------------------------

def _kept_dataset(data, field, copy):
    """What advection keeps of a step: a dataset of the step's topology (by
    reference) with copies of its positions and velocity (``_Step``), at its
    origin, spatial unit and time; piece by piece for pieces."""
    if isinstance(data, PartitionedDataSet):
        return PartitionedDataSet([_kept_dataset(piece, field, copy) for piece in data],
                                  world=data.world)
    step = _Step(data, [field], positions=True, copy=copy)
    return DataSet(step.topology, step.geometry, fields=dict(step.fields), **data._frame())


def _frame_of(data, check=None):
    """A dataset's, or the first piece's, ``_frame()`` and device; over pieces, on a rank
    with none, the others' frame on the default device (``_stand_in``: collective),
    with ``check`` agreed on (``common_origin``)."""
    if not isinstance(data, PartitionedDataSet):
        return data._frame(), data.device
    from tack.dataset.algorithms._common import _stand_in
    from tack.dataset.partitioned import common_origin

    nothing = _stand_in(data, common_origin(data, "advection over pieces", check=check))
    first = nothing if nothing is not None else next(iter(data))
    return first._frame(), first.device


def _velocities(data, field, x, *others):
    """The velocity at world points ``x`` (``(n, 3)`` f64), the values of the fields
    ``others`` there, and whether a cell holds each point: ``probe``, over pieces
    routed to the rank whose cells hold them. Over pieces the probe is
    collective, so a rank with no points probes too."""
    from tack.dataset.algorithms.probe import probe

    names = (field, *others)
    if not len(x) and not isinstance(data, PartitionedDataSet):
        return (*(np.zeros((0, 3)) for _ in names), np.zeros(0, bool))
    out = probe(data, x, fields=list(names))
    if not len(x):
        return (*(np.zeros((0, 3)) for _ in names), np.zeros(0, bool))
    return (*(np.asarray(out.fields[n].values.numpy(), np.float64).reshape(len(x), -1)
              for n in names), out.fields["valid"].values.numpy() == 1)


#: The previous step's velocity, beside this one's on one mesh (``_with_before``).
_BEFORE = "__tack_before__"


def _alike(a, b):
    """Whether fields ``a`` and ``b`` of two steps on one mesh hold their values
    alike -- spaces of one class and parameters (an array parameter, the same
    array) that store their values -- so that ``b``'s space holds ``a``'s values.
    The velocity may be on the cells one step and on the points the next."""
    from tack.dataset.spaces import _is_array

    s, t = a.space, b.space
    if type(s) is not type(t) or s.evaluated or s.params.keys() != t.params.keys():
        return False
    for name, x in s.params.items():
        y = t.params[name]
        if not (x is y or (not (_is_array(x) or _is_array(y)) and x == y)):
            return False
    return True


def _with_before(before, now, field):
    """``now`` with ``before``'s velocity beside its own (``_BEFORE``), both steps
    on one mesh, so that one probe gives both; piece by piece over pieces. ``None``
    where any piece's two velocities are not ``_alike`` (collective)."""
    if isinstance(now, PartitionedDataSet):
        if not all(now.world._allgather(all(_alike(b.fields[field], n.fields[field])
                                            for b, n in zip(before, now)))):
            return None
        return PartitionedDataSet([_with_before(b, n, field) for b, n in zip(before, now)],
                                  world=now.world)
    velocity, earlier = now.fields[field], before.fields[field]
    fields = {field: velocity, _BEFORE: Field(velocity.space, earlier.values, earlier.unit)}
    return DataSet(now.topology, now.geometry, fields=fields, ghosts=now.ghosts,
                   **now._frame())


class Advection:
    """Particles carried by ``field`` (a velocity field's name) through the steps:
    each ``update(data)`` moves every particle from its time to the step's, with
    the velocity linear in time between the previous step (kept, as a copy unless
    ``copy`` is false) and this one, and returns the particles at their current
    points -- a point cloud at the step's time with ``seed`` (each particle's
    number), ``source`` (the seed point it came from, or -1), ``release`` (its
    release time), ``time`` and ``status`` (``ParticleStatus`` bits, as
    ``advect``'s).

    ``seeds`` (an ``(n, 3)`` array of world positions) are released at the first
    step's time; ``inject(points, time=None)`` adds particles at any later time
    (by default the last step's). The velocity is the probe of each step's own
    mesh, so the mesh may change between steps, and over a ``PartitionedDataSet``
    each point is probed on the rank whose cells hold it: the particles stay on
    the rank that released them. ``integrator`` is ``"rk4"`` or ``"euler"``;
    ``step_size`` (in the steps' time unit) bounds each substep, by default one
    substep from step to step. A particle leaving the mesh stops where it last
    was, marked ``SPATIAL_BOUNDS``; one released outside has ``SUCCESS``
    cleared. Times convert between the steps' unit and the velocity's (a
    velocity in m/s over steps in minutes).

    Two steps on one mesh -- the same topology, or one of the same arrays and
    positions, as a reader gives each step -- are integrated in one launch of
    ``advect``'s kernel, the velocity blended in time between them, on the
    steps' device. Over pieces on one mesh, each particle is integrated so by
    the piece whose cells hold it, wherever its rank, until a step would leave
    that piece; that step is taken on the host, and the piece it ends in goes
    on. Steps on different meshes are integrated on the host, probing each
    step's mesh at every stage; every way gives the same paths."""

    def __init__(self, field, seeds=None, integrator="rk4", step_size=None, copy=True):
        if integrator not in ("rk4", "euler"):
            raise ValueError(f"integrator is 'rk4' or 'euler', not {integrator!r}")
        if step_size is not None and not step_size > 0:
            raise ValueError(f"step_size must be positive, not {step_size!r}")
        self.field, self.integrator = field, integrator
        self.step_size, self.copy = step_size, copy
        self.x = np.zeros((0, 3))
        self.time = np.zeros(0)
        self.release = np.zeros(0)
        self.seed = np.zeros(0, np.int64)
        self.source = np.zeros(0, np.int64)
        self.status = np.zeros(0, np.int32)
        self._pending = []                       # (points, time or None, sources)
        self._previous = None                    # the kept previous step
        self._last = None                        # its time, in the stream's unit
        self._unit = None                        # the stream's time unit
        self._released = 0
        if seeds is not None:
            points = np.asarray(seeds, np.float64).reshape(-1, 3)
            self.inject(points, None, np.arange(len(points)))

    def inject(self, points, time=None, sources=None):
        """Release particles at world ``points`` at ``time`` (in the steps' time
        unit; by default the last step's, or the first step's before any); the next
        update carries each from its release time. ``sources`` numbers the seed
        point each came from (``-1`` by default). A time before the last step is
        refused: that interval has been integrated."""
        points = np.asarray(points, np.float64).reshape(-1, 3)
        if time is not None and self._last is not None and time < self._last:
            raise ValueError(f"particles released at {time}, before the last step's time "
                             f"{self._last}: that interval is integrated")
        sources = np.full(len(points), -1, np.int64) if sources is None else \
            np.asarray(sources, np.int64).reshape(-1)
        if len(sources) != len(points):
            raise ValueError("one source per point")
        self._pending.append((points, None if time is None else float(time), sources))

    def _stream_time(self, data):
        """``data``'s time in the stream's time unit (the first step's). Over pieces,
        a piece without the velocity is refused on every rank."""
        from tack.dataset.partitioned import fields_of

        frame, _ = _frame_of(data, check=lambda: fields_of(data, self.field, "advection"))
        if frame["time"] is None:
            raise ValueError("advection needs each step's time (DataSet.time)")
        if self._last is None:
            self._unit = frame["time_unit"]
            return frame["time"]
        unit = frame["time_unit"]
        if (unit is None) != (self._unit is None):
            raise ValueError("the steps' times change from a unit to none, or back")
        if unit is not None and unit != self._unit:
            factor, offset = units.conversion(unit, self._unit)
            return frame["time"] * factor + offset
        return frame["time"]

    def _velocity_scale(self, data):
        """Steps' time unit to the velocity's: the factor to multiply a time by."""
        from tack.dataset.algorithms._common import _time_unit

        if self._unit is None:
            return 1.0
        velocity_time = _time_unit(data, self.field)
        if velocity_time is None:
            return 1.0
        return units.conversion(self._unit, velocity_time)[0]

    def _release(self, data, t1):
        """Particles pending with a release time at or before ``t1``."""
        keep, due = [], []
        for points, time, sources in self._pending:
            time = (self._last if self._last is not None else t1) if time is None else time
            if time > t1:
                keep.append((points, time, sources))
                continue
            if self._last is None and time < t1:
                raise ValueError(f"particles released at {time}, before the first step's "
                                 f"time {t1}")
            due.append((points, time, sources))
        # Released outside any cell: SUCCESS cleared, as advect's seeds. Those
        # released before t1 are probed on the previous step, the rest on this one,
        # each group at once: over pieces every rank probes both, its points or none,
        # since the probe is collective (a rank with fewer releases waited forever).
        earlier = [self._previous is not None and time < t1 for _, time, _ in due]
        inside = [None] * len(due)
        for where, which in ((self._previous, True), (data, False)):
            group = [i for i, e in enumerate(earlier) if e == which]
            if where is None or not (group or isinstance(data, PartitionedDataSet)):
                continue
            points = np.concatenate([due[i][0] for i in group]) if group else \
                np.zeros((0, 3))
            _, found = _velocities(where, self.field, points)
            for i, part in zip(group, np.split(found, np.cumsum(
                    [len(due[i][0]) for i in group])[:-1]) if group else []):
                inside[i] = part
        for (points, time, sources), found in zip(due, inside):
            n = len(points)
            ids = self._released + np.arange(n, dtype=np.int64)
            if isinstance(data, PartitionedDataSet):
                ids += np.int64(data.world.rank) << np.int64(40)
            self._released += n
            status = np.where(found, 1, 0).astype(np.int32)
            for name, values in (("x", points), ("time", np.full(n, time)),
                                 ("release", np.full(n, time)), ("seed", ids),
                                 ("source", sources), ("status", status)):
                setattr(self, name, np.concatenate([getattr(self, name), values]))
        self._pending = keep

    def update(self, data):
        """Carry the particles to ``data``'s time and return them (see the class)."""
        t1 = self._stream_time(data)
        if self._last is not None and not t1 > self._last:
            raise ValueError(f"the steps' times do not increase: {self._last} then {t1}")
        self._release(data, t1)
        if self._previous is not None:
            self._advance(self._previous, data, self._last, t1)
        self._previous = _kept_dataset(data, self.field, self.copy)
        self._last = t1
        return self._particles(data)

    def _moving(self, t1):
        """The particles still to carry to ``t1``."""
        return np.flatnonzero(((self.status & 1) != 0) & ((self.status & 2) == 0)
                              & (self.time < t1 - 1e-12 * max(abs(t1), 1.0)))

    def _advance(self, before, now, t0, t1):
        if not _one_mesh(before, now):
            self._advance_on_host(before, now, t0, t1)
        elif isinstance(now, PartitionedDataSet):
            self._advance_pieces(before, now, t0, t1)
        else:
            self._advance_on_device(before, now, t0, t1)

    def _carry(self, before, now, x, time, status, t0, t1, scale, piece=False):
        """The advection kernel on one mesh (``now``'s, ``before`` the previous
        step's velocity on it): particles at world points ``x`` carried from their
        ``time``s towards ``t1``, with the velocity blended in time. Returns their
        points relative to ``now``'s origin, their times and the kernel's status,
        ``SPATIAL_BOUNDS`` (4) where a step would have left the mesh, the particle
        where that step began. A ``piece``'s ghost cells are outside it: their
        owner's velocity, which a ghost's copy may not match, decides a step
        reaching one."""
        from tack.dataset.algorithms._advection import _AT_END, _run

        n = len(x)
        # The kernel's times are from t0, in the velocity's time unit.
        limit = self.step_size if self.step_size is not None else t1 - t0
        with now.device:
            xs, times, _, out, *_ = _run(
                now, self.field, x, limit * scale, 1 << 30, self.integrator, None, False,
                resume=((time - t0) * scale, np.zeros(n, np.int32), status),
                before=before, until=(t1 - t0) * scale, handoff=piece)
        out = out.numpy().astype(np.int32)
        ended = (out & int(_AT_END)) != 0
        times = np.where(ended, t1, t0 + times.numpy().astype(np.float64) / scale)
        return (xs.numpy().astype(np.float64).reshape(n, 3), times,
                out & ~int(_AT_END))

    def _advance_on_device(self, before, now, t0, t1):
        """Both steps on one mesh: the advection kernel, with the velocity blended
        in time, carries every moving particle from its time to ``t1`` in one
        run (Viskores' and VTK's way of keeping the loop off the host)."""
        moving = self._moving(t1)
        if not len(moving):
            return
        xs, times, status = self._carry(before, now, self.x[moving], self.time[moving],
                                        self.status[moving], t0, t1,
                                        self._velocity_scale(now))
        self.x[moving] = xs + np.asarray(now.origin)
        self.time[moving] = times
        # A particle that would have left stops where it was, as on the host.
        self.status[moving] = np.where((status & 4) != 0, status | 2, status)

    def _advance_pieces(self, before, now, t0, t1):
        """Pieces on one mesh, by rounds until no particle moves on any rank: each
        moving particle goes to the piece one of whose own cells holds it
        (``tack.communication.Routing``), whose kernel carries it until it reaches
        ``t1`` or a step would leave the piece, and comes back to its rank. That
        step is taken on the host (``_substep``), probing every piece: it ends in
        another piece, which carries it on in the next round, or it leaves the
        mesh and the particle stops where it was. Every round each particle moving
        takes a step or stops, so the rounds end. Collective."""
        from tack import communication as tc
        from tack.dataset.algorithms._common import _stand_in
        from tack.dataset.algorithms._particles import _bits, _boxes, _holder, _unbits
        from tack.dataset.locator import cached_locator
        from tack.dataset.partitioned import common_origin

        scale = self._velocity_scale(now)
        origin = common_origin(now, "advection over pieces")
        # A rank with no pieces takes part through a piece of nothing (no box, so
        # no particle goes there) and sends its particles to the pieces.
        nothing = _stand_in(now, origin)
        pairs = list(zip(before, now)) if nothing is None else [(nothing, nothing)]
        pieces = [piece for _, piece in pairs]
        blocks = tc.Blocks([piece.device for piece in pieces], now.world)
        locators = {}
        for g, piece in zip(blocks.local, pieces):
            if piece.num_points:
                with piece.device:
                    locators[g] = cached_locator(piece)
        boxes, holds = _boxes(blocks, pieces, locators), _holder(blocks, pieces, locators)
        home = blocks.local.start                   # this rank's particles' block
        both = _with_before(before, now, self.field)
        velocity = self._blended(before, now, t0, t1) if both is None else \
            self._blended(None, both, t0, t1)
        origin = np.asarray(origin, np.float64)
        like = {"x0": tack.u64, "x1": tack.u64, "x2": tack.u64, "time": tack.u64,
                "status": tack.i32}
        while True:
            moving = self._moving(t1)
            if not blocks.all_reduce({g: len(moving) if g == home else 0
                                      for g in blocks.local}):
                break
            routing = tc.Routing(blocks, {g: self.x[moving] - origin if g == home
                                          else np.zeros((0, 3)) for g in blocks.local},
                                 boxes, holds)
            where = routing.destination[home]
            # No piece holds it (coming within the probe's tolerance, not the
            # locators'): it has left, where it is.
            self.status[moving[where < 0]] |= 2 | 4
            device = blocks.device(home)
            sent = {g: {} for g in blocks.local}
            sent[home] = {"time": tack.array(self.time[moving].view(np.uint64), tack.u64,
                                             device=device),
                          "status": tack.array(self.status[moving], tack.i32, device=device)}
            arrived = routing.send(sent)
            answers = {g: {} for g in blocks.local}
            for g, (kept, piece) in zip(blocks.local, pairs):
                x, values = arrived[g]
                if not len(x):
                    continue
                xs, times, status = self._carry(
                    kept, piece, x + origin, values["time"].numpy().view(np.float64),
                    values["status"].numpy(), t0, t1, scale, piece=True)
                device = blocks.device(g)
                answers[g] = {**_bits(xs, device),
                              "time": tack.array(times.view(np.uint64), tack.u64,
                                                 device=device),
                              "status": tack.array(status, tack.i32, device=device)}
            back = routing.reply(answers, like)[home]
            taken, rows = moving[where >= 0], where >= 0
            self.x[taken] = _unbits(back)[rows] + origin
            self.time[taken] = back["time"].numpy().view(np.float64)[rows]
            status = back["status"].numpy()[rows]
            self.status[taken] = status & ~4
            self._substep(velocity, taken[(status & 4) != 0], t0, t1, scale)

    def _advance_on_host(self, before, now, t0, t1):
        """Each step's velocity probed in its own mesh at every stage: steps on
        different meshes."""
        scale = self._velocity_scale(now)
        # Over pieces every rank takes each substep, moving particles or not, until
        # none moves on any: the probes are collective.
        world = now.world if isinstance(now, PartitionedDataSet) else None
        velocity = self._blended(before, now, t0, t1)
        while True:
            moving = self._moving(t1)
            if not (len(moving) if world is None
                    else any(world._allgather(bool(len(moving))))):
                break
            self._substep(velocity, moving, t0, t1, scale)

    def _blended(self, before, now, t0, t1):
        """The velocity at world points and times, linear in time from ``t0`` to
        ``t1``, and whether the mesh holds each point: each step's probed in its
        own mesh, or, ``before`` ``None``, both at once from ``now``
        (``_with_before``: one mesh)."""
        def velocity(x, t):
            a = np.clip((t - t0) / (t1 - t0), 0.0, 1.0)[:, None]
            if before is None:
                v1, v0, inside = _velocities(now, self.field, x, _BEFORE)
            else:
                v0, in0 = _velocities(before, self.field, x)
                v1, in1 = _velocities(now, self.field, x)
                inside = in0 & in1
            return (1.0 - a) * v0 + a * v1, inside

        return velocity

    def _substep(self, velocity, rows, t0, t1, scale):
        """One substep of particles ``rows`` on the host, ``velocity`` (``_blended``)
        probed at every stage: those whose step stays in the mesh move, the
        others stop where they were. Over pieces the probes are collective:
        every rank takes it, with particles or none."""
        limit = self.step_size if self.step_size is not None else t1 - t0
        x, t = self.x[rows], self.time[rows]
        h = np.minimum(limit, t1 - t)
        k1, ok = velocity(x, t)
        if self.integrator == "euler":
            step = k1
        else:
            k2, ok2 = velocity(x + 0.5 * (h * scale)[:, None] * k1, t + 0.5 * h)
            k3, ok3 = velocity(x + 0.5 * (h * scale)[:, None] * k2, t + 0.5 * h)
            k4, ok4 = velocity(x + (h * scale)[:, None] * k3, t + h)
            ok = ok & ok2 & ok3 & ok4
            step = (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        end = x + (h * scale)[:, None] * step
        _, inside = velocity(end, t + h)
        ok &= inside
        # A particle whose step leaves the mesh stops where it was.
        left = rows[~ok]
        self.status[left] |= 2 | 4
        went = rows[ok]
        self.x[went] = end[ok]
        self.time[went] = (t + h)[ok]
        self.status[went] |= 16

    def _particles(self, data):
        from tack.dataset.dataset import point_cloud

        frame, device = _frame_of(data)
        fields = {"seed": self.seed, "source": self.source, "release": self.release,
                  "time": self.time, "status": self.status}
        time_units = {"release": self._unit, "time": self._unit} if self._unit else {}
        cloud = point_cloud(self.x, fields=fields, origin=frame["origin"], device=device,
                            unit=frame["unit"], units=time_units)
        cloud.time, cloud.time_unit = frame["time"], frame["time_unit"]
        if isinstance(data, PartitionedDataSet):
            return PartitionedDataSet([cloud], world=data.world)
        return cloud


def _rows(particles):
    """A particle cloud's (or a piece list's) world positions and fields, host."""
    pieces = list(particles) if isinstance(particles, PartitionedDataSet) else [particles]
    x = np.concatenate([p.world_positions() for p in pieces]) if pieces else np.zeros((0, 3))
    get = lambda name: np.concatenate([np.asarray(p.fields[name].values.numpy())
                                       for p in pieces]) if pieces else np.zeros(0)
    return x, {name: get(name) for name in ("seed", "time", "status")}, pieces


def _lines(groups, x, frame, like, point_fields):
    """A dataset of the world points ``x`` (sorted by group) and a polyline through
    each group of two or more; a group of one point keeps its point, in no line."""
    from tack.dataset.polyhedra import PolylineTopology
    from tack.dataset.spaces import H1, Values

    _, starts, counts = np.unique(groups, return_index=True, return_counts=True)
    lined = counts >= 2
    offsets = np.concatenate([[0], np.cumsum(counts[lined])]).astype(np.int64)
    members = np.concatenate([np.arange(a, a + c) for a, c in
                              zip(starts[lined], counts[lined])]) if lined.any() else \
        np.zeros(0, np.int64)
    with like.device:
        topology = PolylineTopology(offsets, members.astype(np.int64), num_points=len(x))
        origin = np.asarray(frame["origin"])
        out = DataSet(topology, (x - origin).astype(like.dtype.numpy_dtype), dtype=like.dtype,
                      **frame)
        for name, (values, unit) in point_fields.items():
            kind = values.dtype.kind
            stored = values.astype(out.dtype.numpy_dtype if kind == "f" else np.int64)
            out.fields[name] = Field(H1(out) if kind == "f" else Values(out, "points"),
                                     tack.array(stored), unit)
    return out


def extend_paths(paths, particles):
    """``paths`` (``None``, or what this returned before) with ``particles`` -- an
    ``Advection``'s output -- appended: each particle's points in time (``seed``
    and ``time`` on the points), and a polyline through those of each particle
    with two or more. Each step adds the particles that moved since (a
    stopped one adds nothing). It stands alone, so particles written step by
    step can be joined into paths afterwards."""
    x, rows, pieces = _rows(particles)
    seed, time = rows["seed"].astype(np.int64), rows["time"].astype(np.float64)
    if paths is not None:
        old_seed = paths.fields["seed"].values.numpy().astype(np.int64)
        old_time = paths.fields["time"].values.numpy().astype(np.float64)
        # A particle's point is new when it is later than its path's last.
        order = np.lexsort((old_time, old_seed))
        last_of = dict(zip(old_seed[order].tolist(), old_time[order].tolist()))
        new = np.array([t_ > last_of.get(s_, -np.inf)
                        for s_, t_ in zip(seed.tolist(), time.tolist())], bool)
        x = np.concatenate([paths.world_positions(), x[new]])
        seed = np.concatenate([old_seed, seed[new]])
        time = np.concatenate([old_time, time[new]])
    order = np.lexsort((time, seed))
    like = pieces[0]
    unit = like.fields["time"].unit
    return _lines(seed[order], x[order], like._frame(), like,
                  {"seed": (seed[order], None), "time": (time[order], unit)})


class Pathlines:
    """Each particle's path through the steps: ``Advection`` (its arguments) and
    ``extend_paths``, each ``update(data)`` returning the current pathlines -- a
    polyline per particle (``seed`` and ``time`` on the points), a particle's
    first point alone until it has moved.
    ``inject`` releases particles as ``Advection.inject`` does."""

    def __init__(self, field, seeds=None, integrator="rk4", step_size=None, copy=True):
        self.advection = Advection(field, seeds, integrator, step_size, copy)
        self.paths = None

    def inject(self, points, time=None):
        self.advection.inject(points, time)

    def update(self, data):
        self.paths = extend_paths(self.paths, self.advection.update(data))
        return self.paths


class Streaklines:
    """The lines of particles released at ``seeds`` (world positions) through the
    steps: every update releases ``releases`` particles at each seed, at times
    spread evenly over the interval since the previous step (one at each step's
    time by default), carries them all (``Advection``), and returns a polyline per
    seed (``source`` on the cells) through the particles it released still in the
    mesh, newest (at the seed) first, with ``source``, ``release`` and ``time`` on
    the points; a seed with one particle so far keeps its point, in no line."""

    def __init__(self, field, seeds, releases=1, integrator="rk4", step_size=None,
                 copy=True):
        if int(releases) < 1:
            raise ValueError("releases is at least 1")
        self.seeds = np.asarray(seeds, np.float64).reshape(-1, 3)
        self.releases = int(releases)
        self.advection = Advection(field, None, integrator, step_size, copy)

    def update(self, data):
        adv = self.advection
        t1 = adv._stream_time(data)
        sources = np.arange(len(self.seeds))
        if adv._last is None:
            adv.inject(self.seeds, t1, sources)
        else:
            for j in range(1, self.releases + 1):
                adv.inject(self.seeds, adv._last + (t1 - adv._last) * j / self.releases,
                           sources)
        particles = adv.update(data)
        x, _, pieces = _rows(particles)
        alive = ((adv.status & 1) != 0) & ((adv.status & 4) == 0) & (adv.source >= 0)
        x, source = x[alive], adv.source[alive]
        release, time = adv.release[alive], adv.time[alive]
        order = np.lexsort((-release, source))          # newest, at the seed, first
        like = pieces[0]
        unit = like.fields["time"].unit
        return _lines(source[order], x[order], like._frame(), like,
                      {"source": (source[order], None), "release": (release[order], unit),
                       "time": (time[order], unit)})

