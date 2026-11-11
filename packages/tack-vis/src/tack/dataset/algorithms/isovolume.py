# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""``isovolume``: the part of ``data`` where ``by`` -- point data (a field or its name) or an implicit function (``tack.dataset.implicit``) -- lies between ``lower`` and ``upper``: ParaView's Iso Volume, as a clip at each end of the range; a high-order field cut through subdivision, its crossings at the field's roots at both ends."""

import numpy as np

import tack
from tack import arrays
from tack.dataset.algorithms._common import _field, _in_unit_of
from tack.dataset.algorithms._refine import _masks
from tack.dataset.algorithms.clip import clip
from tack.dataset.algorithms.implicit_values import implicit_values
from tack.dataset.dataset import DataSet, Field
from tack.runtime.dispatch import on_inputs_device

__all__ = ['isovolume']

#: The name the range's field takes in a dataset that does not hold it (an
#: implicit function's values, or a field given as a ``Field`` of another name).
_RANGE = "__tack_isovolume__"


@on_inputs_device
def isovolume(data, by, lower, upper, fields=None, subdivision=None, tolerance=None,
              exact=True):
    """The part of ``data`` where ``by`` -- point data (a field or its name) or an
    implicit function (``tack.dataset.implicit``) -- lies in ``[lower, upper]``:
    ParaView's Iso Volume. It is ``clip`` at ``lower`` and then, of what that keeps,
    ``clip`` below ``upper`` (``inside``), so cells wholly in the range keep their
    shape, cut cells become the pieces the clip tables list at both ends, and point
    data is interpolated onto the new points, cell data goes to each piece. A point
    at either end is kept. Polyhedra are clipped into polyhedra, as ``clip`` does.

    A curved geometry or a field above order 1 (or ``subdivision`` above 1, or a
    ``tolerance``) is cut through subdivision, as ``clip`` cuts it, and the result
    is ``PartitionedDataSet([whole, part])``: ``whole`` the cells the field's
    Bernstein bound keeps entirely within the range, as they are (curved, their
    fields of any order), and ``part`` the cut of the others that the bound does not
    drop, in ``subdivision`` lattice steps (by default the largest order of the
    field and the geometry; with ``tolerance``, in the field's unit, the fewest
    whose estimated error is within it). With ``exact`` (the default) each crossing
    on a sub-cell's edge is where the field takes the end's value along it, at both
    ends, and the fields are evaluated there; with ``exact=False``, where the line
    between the lattice values does. The second end's crossings are found along the
    sub-edges the first clip left, between a lattice point and the other or the
    first end's crossing; one on an edge to a clip's centroid point (an average of
    crossings, which a hexahedron's, wedge's or pyramid's case may add) is on the
    straight line, as a single clip's centroid is."""
    from tack.dataset.algorithms.subdivide import split_by_range, subdivides
    from tack.dataset.ghosts import visible
    from tack.dataset.polyhedra import PolygonalTopology, PolyhedralTopology

    polyhedral = isinstance(data.topology, PolyhedralTopology) and \
        not isinstance(data.topology, PolygonalTopology)
    if not polyhedral and not getattr(data.topology, "reference_cells", True):
        raise NotImplementedError("isovolume takes cells of the linear shapes or polyhedra, "
                                  "not polygons or polylines")
    implicit = hasattr(by, "value") and not isinstance(by, (str, Field))
    field = implicit_values(data, by) if implicit else _field(data, by)
    if arrays.width_of(field.values):
        raise TypeError("isovolume needs a scalar field")
    lower = _in_unit_of(field, lower, "isovolume's lower")
    upper = _in_unit_of(field, upper, "isovolume's upper")
    if not lower <= upper:
        raise ValueError(f"isovolume's lower ({lower}) is above its upper ({upper})")
    if tolerance is not None:
        tolerance = _in_unit_of(field, tolerance, "isovolume's tolerance")
    name = next((k for k, f in data.fields.items() if f is field), None)
    kept = fields
    if name is None:
        # The range's field joins the dataset for the clips, and leaves with them.
        name = _RANGE
        if fields is None:
            kept = list(data.fields)
        data = DataSet(data.topology, data.geometry, fields={**data.fields, name: field},
                       sets=dict(data.sets), ghosts=data.ghosts, global_ids=data.global_ids,
                       **data._frame())
    with_range = None if kept is None else [*kept, name]
    refines = not polyhedral and (subdivides(data, field) or tolerance is not None
                                  or (subdivision is not None and subdivision > 1))
    if not refines:
        first = clip(data, name, lower, False, with_range)
        return clip(first, name, upper, True, kept)
    if _masks(data)[1]:
        data = visible(data)
    field = data.fields[name]
    levels = tack.array(np.asarray([lower, upper], arrays.dtype_of(field.values).numpy_dtype))

    def both_ends(refined, cut, carried):
        from tack.dataset.algorithms._crossings import Chained

        first = clip(refined, cut, lower, False, None if carried is None else [*carried, cut])
        crossings = getattr(refined, "_crossings", None)
        if crossings is not None and crossings.placed is not None:
            first._crossings = Chained(crossings, crossings.placed, refined, first, cut,
                                       upper)
        return clip(first, cut, upper, True, carried)

    return split_by_range(data, field, lower, upper, subdivision, tolerance, kept, both_ends,
                          levels if exact else None)
