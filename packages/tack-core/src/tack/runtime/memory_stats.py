# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The memory Tack's own arrays hold, counted per device, with a peak (private).

Every array Tack makes is counted where it is allocated (`tack.empty`'s
``_allocate`` and ``_empty_for_mpi``, which every backend's ``allocate_array``
goes through), and its bytes are given back when its buffer is collected. So
the CPU, Metal, CUDA, HIP, Level Zero and Warp are counted alike, which the
process's allocator statistics (the CPU only) and the drivers (behind CUDA's
pool, HIP's buffer cache and Metal's pool) cannot do. What is counted:

- the bytes an array asked for (``DeviceBuffer.nbytes``), not what a pool
  rounded them up to;
- memory from outside (``wrap_ptr``, ``array_from_ptr``, a DLPack or NumPy
  import without a copy) is not Tack's, and is not counted;
- a view shares its array's buffer, and is counted once, with the buffer: the
  memory is held while any view of it is;
- a block a backend's cache hands to a new array is counted while that array
  holds it, not while it waits in the cache (``release_memory`` empties the
  caches);
- the few bytes of a compiled kernel's scalar packs are not counted.

`reset_peak` makes the peak what is live, so a test measures one call:
``reset_peak(d); start = live(d); f(); peak(d) - start``. Each function takes a
`tack.Device`, a spec (``"cuda:1"``), a backend, or None for the default
device. The cut filters' memory plan (``docs/design/plans/cut-filters-memory.md``,
D1) is why it exists; spilling to host memory under a budget (#387) will read
it too.
"""

import threading
import weakref

# A backend's count, kept while the backend lives. What a buffer gives back is
# held with its weak reference, the count and not the backend, so a buffer
# never keeps its backend alive here.
_counts = weakref.WeakKeyDictionary()
# Each counted buffer's weak reference: (count, bytes). A weak reference with a
# callback costs a third of `weakref.finalize` (0.2 against 0.8 us), and every
# array pays it.
_held = {}
# Reentrant: a buffer's finalizer can run inside a count's update, when the
# update's allocation sets off the garbage collector.
_lock = threading.RLock()


class _Count:
    __slots__ = ("live", "peak")

    def __init__(self):
        self.live = 0
        self.peak = 0


def _count_of(backend):
    count = _counts.get(backend)
    if count is None:
        with _lock:
            count = _counts.setdefault(backend, _Count())
    return count


def _released(ref):
    count, nbytes = _held.pop(ref)
    with _lock:
        count.live -= nbytes


def counted(backend, buffer):
    """``buffer``, just allocated on ``backend``, added to the backend's count until
    it is collected."""
    nbytes = int(buffer.nbytes)
    count = _count_of(backend)
    with _lock:
        count.live += nbytes
        if count.live > count.peak:
            count.peak = count.live
    _held[weakref.ref(buffer, _released)] = (count, nbytes)
    return buffer


def _backend(device):
    from tack.lang.array import _target

    return _target(device)


def live(device=None) -> int:
    """The bytes Tack's arrays hold on ``device`` now."""
    return _count_of(_backend(device)).live


def peak(device=None) -> int:
    """The most bytes Tack's arrays have held on ``device`` at once since it was
    first used or `reset_peak` was called."""
    return _count_of(_backend(device)).peak


def reset_peak(device=None) -> int:
    """Make ``device``'s peak what is live now, and return it."""
    count = _count_of(_backend(device))
    with _lock:
        count.peak = count.live
        return count.peak
