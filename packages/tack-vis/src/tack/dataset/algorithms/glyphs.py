# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""``glyphs``: an arrow along a vector field at each point, or at each cell's centre,
VTK's vtkGlyph3D with its arrow source."""

import math

import tack
from tack import arrays
from tack.dataset import ids as id_types
from tack.dataset import shapes
from tack.dataset.algorithms._common import _field
from tack.dataset.algorithms.tubes import _across
from tack.dataset.carry import Take, carry
from tack.dataset.dataset import Field
from tack.dataset.spaces import H1, L2, Constant, Values
from tack.dataset.topology import UnstructuredTopology, uniform_cells
from tack.runtime.dispatch import on_inputs_device

__all__ = ['glyphs']

#: An arrow of length 1: the shaft's radius and length, the head's radius.
SHAFT_RADIUS = tack.constant(0.03)
SHAFT_LENGTH = tack.constant(0.65)
HEAD_RADIUS = tack.constant(0.1)


@tack.kernel
def _arrows(positions, vectors, scale, by, scales, res, step, out, source, connectivity):
    # Point i's arrow: a shaft (two rings), the head's base ring, its tip and the
    # shaft's base centre, 3 res + 2 points; as triangles, the shaft's sides, the
    # head's base, its cone and the shaft's base, 6 res, wound outward.
    for i in range(len(positions)):
        v = vectors[i]
        length = v.norm()
        t = tack.vec3(0.0, 0.0, 1.0)
        if length > 0.0:
            t = v / length
        size = length * scale
        if by == 1:
            size = scales[i] * scale
        n = _across(t)
        b = t.cross(n)
        p = positions[i]
        first = i * (3 * res + 2)
        for s in range(res):
            angle = step * s
            d = tack.cos(angle) * n + tack.sin(angle) * b
            out[first + s] = p + size * SHAFT_RADIUS * d
            out[first + res + s] = p + size * (SHAFT_LENGTH * t + SHAFT_RADIUS * d)
            out[first + 2 * res + s] = p + size * (SHAFT_LENGTH * t + HEAD_RADIUS * d)
        out[first + 3 * res] = p + size * t
        out[first + 3 * res + 1] = p
        for k in range(3 * res + 2):
            source[first + k] = i
        tip = first + 3 * res
        centre = tip + 1
        tri = i * 6 * res
        for s in range(res):
            nxt = (s + 1) % res
            c = 3 * (tri + 6 * s)
            connectivity[c] = first + s                       # shaft side
            connectivity[c + 1] = first + nxt
            connectivity[c + 2] = first + res + nxt
            connectivity[c + 3] = first + s
            connectivity[c + 4] = first + res + nxt
            connectivity[c + 5] = first + res + s
            connectivity[c + 6] = first + res + s             # the head's base, facing back
            connectivity[c + 7] = first + res + nxt
            connectivity[c + 8] = first + 2 * res + nxt
            connectivity[c + 9] = first + res + s
            connectivity[c + 10] = first + 2 * res + nxt
            connectivity[c + 11] = first + 2 * res + s
            connectivity[c + 12] = first + 2 * res + s        # cone
            connectivity[c + 13] = first + 2 * res + nxt
            connectivity[c + 14] = tip
            connectivity[c + 15] = first + nxt                # the shaft's base
            connectivity[c + 16] = first + s
            connectivity[c + 17] = centre


@tack.kernel
def _used_points(cells, marks, used):
    # The points of each cell that is not hidden or refined.
    for c in cells:
        e = cells.entity_id(c)
        if (tack.i32(marks[e]) & 40) == 0:
            for j in range(cells.NUM_POINTS):
                tack.atomic_max(used, cells.point_id(c, j), 1)


@tack.kernel
def _from_kept(kept, source):
    # An arrow point's source, a kept point's number, as the point's own id.
    for i in range(source.shape[0]):
        source[i] = kept[source[i]]


def _at_points(field):
    """Whether ``field`` has one value at each point, its first values: ``H1`` of any
    order numbered by entity, values on points."""
    space = field.space
    return (isinstance(space, H1) and space.dofs is None) or \
        (isinstance(space, Values) and space.on == "points")


def _cell_wise(field):
    """Whether ``field`` has a value per cell, not per point: ``L2`` (a DG field, whose
    cells sharing a point each have their own value there), ``Constant``, values on
    cells."""
    space = field.space
    return isinstance(space, (L2, Constant)) or (isinstance(space, Values)
                                                 and space.on == "cells")


@on_inputs_device
def glyphs(data, vectors, scale=1.0, scale_by=None, resolution=8, fields=None, at=None):
    """An arrow along ``vectors`` (a 3-vector field, or its name) at each of ``data``'s
    points, or with ``at="cells"`` at each cell's centre, as VTK's ``vtkGlyph3D``
    places its arrow source: from the place toward the vector, ``scale`` times its
    length long, or ``scale`` times ``scale_by`` (a scalar field, or its name) where
    given. An arrow of length 1 has a shaft 0.65 long and 0.03 in radius and a head
    0.1 in radius, ``resolution`` points round, as triangles wound outward. A zero
    vector gives an arrow of no size.

    How many arrows there are follows the mesh, not the field's order. At the
    points (the default for point fields), an ``H1`` field of any order and a
    curved geometry give their values there, which are their stored corner
    values, and the arrows' points carry the point fields; cell fields are left
    behind. At the cells' centres -- where the geometry maps each cell's
    parametric centre (``cell_centers``) -- every field is evaluated through its
    basis (``values_at_centers``), and the arrows carry those values. A field
    with a value per cell, ``L2`` (DG) above all, is glyphed there, the default
    for it: a point several such cells share has a value in each, so ``at="points"``
    refuses it (``project`` it into ``H1`` to average them).

    Glyph a subset (``mask_points``) or a slice to see anything in a dense field,
    and other places by probing first (``glyphs(probe(data, points), ...)``).
    Points only hidden or refined cells use have no arrow, nor have those cells."""
    from tack.dataset.algorithms._refine import _masks
    from tack.dataset.ghosts import visible

    field = _field(data, vectors)
    sizes = None if scale_by is None else _field(data, scale_by)
    if at is None:
        at = "cells" if _cell_wise(field) or (sizes is not None and _cell_wise(sizes)) \
            else "points"
    if at == "cells":
        return _at_cells(data, field, sizes, scale, resolution, fields)
    if at != "points":
        raise ValueError(f"glyphs are at the points or the cells, not {at!r}")
    for name, f in (("vectors", field), ("scale_by", sizes)):
        if f is not None and _cell_wise(f):
            raise TypeError(f"glyphs at the points: {name} is in {f.space!r}, which has a "
                            "value per cell, several where cells share a point; glyph it "
                            "at=\"cells\", or project it into H1 to average them")
    marks, hidden = _masks(data)
    if hidden and not getattr(data.topology, "reference_cells", True):
        data = visible(data)                  # polyhedra and polygons: the cells left
        hidden = 0
    if not _at_points(data.geometry):
        raise TypeError("glyphs at the points needs an H1 geometry, one position a point; "
                        "glyph at=\"cells\"")
    res = int(resolution)
    if res < 3:
        raise ValueError(f"an arrow has a resolution of at least 3, not {res}")
    if arrays.width_of(field.values) != 3 or not _at_points(field):
        raise TypeError("glyphs takes a 3-vector field on the points")
    if sizes is not None and (arrays.width_of(sizes.values) or not _at_points(sizes)):
        raise TypeError("scale_by is a scalar field on the points")
    from tack.dataset.carry import _point_values

    # Each point's position and value: an H1 field of any order (a curved geometry
    # too) holds its points' first.
    n = data.num_points
    positions = _point_values(data.geometry.values, n)
    directions = _point_values(field.values, n)
    by = _point_values(sizes.values, n) if sizes is not None else None
    kept = None
    if hidden:
        # The points some cell that is not hidden or refined uses, in order -- the
        # points the cells left once extracted would have.
        from tack.dataset.algorithms._common import _compact
        from tack.dataset.carry import _point_values, _take

        used = tack.zeros((data.num_points,), tack.i32)
        for group in data.launch_groups("cells", []):
            if group.count:
                _used_points(data.domain_view("cells", group), marks, used)
        _, kept, _ = _compact(used, data.id_dtype)
        positions = _take(arrays.materialize(positions), kept)
        directions = _take(arrays.materialize(directions), kept)
        by = _take(arrays.materialize(by), kept) if by is not None else None
    n = arrays.size_of(positions)
    total = n * (3 * res + 2)
    idt = id_types.at_least(data.id_dtype, total)
    out = tack.empty((total,), tack.vec(3, data.dtype))
    source = tack.empty((total,), data.id_dtype)
    connectivity = tack.empty((n * 18 * res,), idt)
    if n:
        _arrows(positions, directions, float(scale), int(sizes is not None),
                by if by is not None else tack.empty((1,), data.dtype), res,
                2.0 * math.pi / res, out, source, connectivity)
        if kept is not None:
            _from_kept(kept, source)
    types, offsets = uniform_cells(int(shapes.TRIANGLE), n * 6 * res, idt)
    topology = UnstructuredTopology(types, offsets, connectivity, num_points=total)
    return carry(data, topology, points=Take(source),
                 geometry=Field(H1(topology), out, data.geometry.unit), fields=fields)


def _at_cells(data, field, sizes, scale, resolution, fields):
    """``glyphs`` at each visible cell's centre: the centres as a point cloud carrying
    every field evaluated there (``values_at_centers``; fields with no value at a
    centre left out), each point marked as its cell, glyphed at its points."""
    from tack.dataset.algorithms.cell_centers import cell_centers
    from tack.dataset.algorithms.values_at_centers import values_at_centers
    from tack.dataset.carry import selected
    from tack.dataset.dataset import DataSet
    from tack.dataset.ghosts import Ghosts, visible

    data, field, *rest = visible(data, field, *([sizes] if sizes is not None else []))
    sizes = rest[0] if rest else None
    for name, f, width in (("vectors", field, 3), ("scale_by", sizes, 0)):
        if f is not None and (arrays.width_of(f.values) or 0) != width:
            raise TypeError(f"glyphs takes {'a 3-vector' if width else 'a scalar'} field as "
                            f"{name}")
    n = data.num_cells
    cloud = DataSet(UnstructuredTopology(*_vertex_rows(n, data.id_dtype), num_points=n),
                    cell_centers(data).values, dtype=data.dtype, **data._frame())
    for name in selected(data, fields):
        source = data.fields[name]
        try:
            values = values_at_centers(data, source).values
        except (TypeError, NotImplementedError):
            continue                       # no value at a cell's centre (values on faces)
        cloud.fields[name] = Field(Values(cloud, "points"), values, source.unit)
    cloud.fields["__tack_glyph__"] = Field(Values(cloud, "points"),
                                           values_at_centers(data, field).values)
    if sizes is not None:
        cloud.fields["__tack_glyph_size__"] = Field(Values(cloud, "points"),
                                                    values_at_centers(data, sizes).values)
    ghosts = data.ghosts
    if ghosts is not None and ghosts.cells is not None and n:
        # A centre is its cell's: a ghost cell's centre a ghost point.
        marks = tack.empty((n,), tack.u8)
        _point_marks(ghosts.cells, marks)
        cloud.ghosts = Ghosts(points=marks)
    keep = [k for k in cloud.fields if not k.startswith("__tack_glyph")]
    return glyphs(cloud, "__tack_glyph__", scale,
                  "__tack_glyph_size__" if sizes is not None else None, resolution, keep,
                  at="points")


@tack.kernel
def _point_marks(cells, out):
    for i in range(cells.shape[0]):
        out[i] = tack.u8(tack.i32(cells[i]) & 1)          # DUPLICATECELL to DUPLICATEPOINT


def _vertex_rows(n, id_dtype):
    """``(types, offsets, connectivity)`` of ``n`` vertex cells, cell ``i`` on point
    ``i``."""
    types, offsets = uniform_cells(int(shapes.VERTEX), n, id_dtype)
    return types, offsets, tack.arange(n, dtype=id_dtype)
