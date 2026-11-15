# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""``surface_normals``: a surface's polygons wound consistently and its normals at the points and cells -- VTK's vtkPolyDataNormals without splitting."""

import numpy as np

import tack
from tack import arrays
from tack.algorithms.scan import exclusive_scan
from tack.dataset import shapes
from tack.dataset.algorithms._common import _table, _vectors, mesh_kept
from tack.dataset.algorithms._links import _average_links
from tack.dataset.algorithms._refine import _masks
from tack.dataset.carry import Same, _take, carry
from tack.dataset.dataset import Field
from tack.dataset.spaces import Values
from tack.dataset.topology import UnstructuredTopology, corner_layout, polyhedral_corners
from tack.runtime.dispatch import on_inputs_device

__all__ = ['surface_normals']

#: A 2D shape's corners in order round it, and in the other order from its first.
_ROUND = {shapes.Triangle: (0, 1, 2), shapes.Quad: (0, 1, 2, 3), shapes.Pixel: (0, 1, 3, 2)}
_REVERSED = {shapes.Triangle: (0, 2, 1), shapes.Quad: (0, 3, 2, 1), shapes.Pixel: (0, 2, 1, 3)}


@tack.kernel
def _edge_relations(cells, sides, pairs, relations, marks, hidden):
    # Each two-sided edge's cells, and whether they wind it alike (1: one must
    # turn), from its side 1: side 0 walks the edge's row forwards, and a
    # consistent neighbour walks it backwards (reflected). An edge with a hidden
    # or refined cell joins nothing: such a cell is its own region, as it is.
    for c in cells:
        for k in range(cells.num_faces(c)):
            if cells.face_side(c, k) == 1 and not (hidden == 1 and (
                    (tack.i32(marks[cells.entity_id(c)]) & 40) != 0
                    or (tack.i32(marks[sides[cells.face_id(c, k)][0]]) & 40) != 0)):
                f = cells.face_id(c, k)
                pairs[f] = [sides[f][0], cells.entity_id(c)]
                relations[f] = 1 - cells.face_orientation(c, k) % 2


@tack.kernel
def _loop_relations(cells, sides, pairs, relations, marks, hidden):
    # A polygon beside shapes on side 1 of an edge: the other polygon on it walks it
    # the other way, as a mixed topology's polygons all do (consistent). An edge
    # with a hidden or refined cell joins nothing, as in _edge_relations.
    for c in cells:
        for k in range(cells.num_faces(c)):
            if cells.face_side(c, k) == 1 and not (hidden == 1 and (
                    (tack.i32(marks[cells.entity_id(c)]) & 40) != 0
                    or (tack.i32(marks[sides[cells.face_id(c, k)][0]]) & 40) != 0)):
                f = cells.face_id(c, k)
                pairs[f] = [sides[f][0], cells.entity_id(c)]
                relations[f] = 0


@tack.kernel
def _rewound_loops(cells, parents, offsets, connectivity):
    # A polygon that winds against its region's first cell, its loop the other way
    # round from its first point.
    for c in cells:
        e = cells.entity_id(c)
        if parents[e] % 2 == 1:
            n = cells.num_points(c)
            for j in range(n):
                connectivity[offsets[e] + j] = cells.point_id(c, (n - j) % n)


@tack.kernel
def _hook(pairs, relations, parents, changed):
    # parents[c] is 2 * parent + parity (whether c winds against its parent).
    # Each edge joins its cells' trees, the larger root under the smaller with
    # the parity the edge implies; an atomic minimum, so a root's parent and
    # parity are written together and concurrent hooks agree on one of them.
    for f in range(pairs.shape[0]):
        a = pairs[f][0]
        if a >= 0:
            b = pairs[f][1]
            va = parents[a]
            vb = parents[b]
            ra = va // 2
            rb = vb // 2
            if ra != rb:
                parity = (va % 2 + vb % 2 + relations[f]) % 2
                tack.atomic_min(parents, max(ra, rb), 2 * min(ra, rb) + parity)
                if changed[0] == 0:          # once a pass, not an atomic per hook
                    tack.atomic_max(changed, 0, 1)


@tack.kernel
def _jump(parents):
    # Every cell straight to its root, its parity summed along the way.
    for c in range(parents.shape[0]):
        v = parents[c]
        r = v // 2
        parity = v % 2
        while parents[r] // 2 != r:
            v = parents[r]
            parity = (parity + v % 2) % 2
            r = v // 2
        parents[c] = 2 * r + parity


@tack.kernel
def _contradictions(pairs, relations, parents, count):
    for f in range(pairs.shape[0]):
        a = pairs[f][0]
        if a >= 0:
            b = pairs[f][1]
            if (parents[a] % 2 + parents[b] % 2) % 2 != relations[f]:
                tack.atomic_add(count, 0, 1)


@tack.kernel
def _rewound(cells, parents, offsets, order, connectivity):
    # A cell that winds against its region's first cell, its corners the other
    # way round from its first.
    for c in cells:
        e = cells.entity_id(c)
        if parents[e] % 2 == 1:
            at = offsets[e]
            for j in range(cells.NUM_POINTS):
                connectivity[at + j] = cells.point_id(c, order[j])


@tack.kernel
def _shape_normals(cells, order, out, areas, marks, hidden):
    # Newell's normal round the corners, of unit length (zero for no area), and its
    # length: twice the area; zero for a hidden or refined cell, which adds nothing
    # at its points.
    for c in cells:
        size = order.shape[0]
        origin = cells.corner_position(c, order[0])
        n = tack.vec3(0.0, 0.0, 0.0)
        for k in range(size):
            n += (cells.corner_position(c, order[k]) - origin).cross(
                cells.corner_position(c, order[k + 1 if k + 1 < size else 0]) - origin)
        length = n.norm()
        if hidden == 1 and (tack.i32(marks[cells.entity_id(c)]) & 40) != 0:
            n = n - n
            length = length - length
        out[cells.entity_id(c)] = n / length if length > 0.0 else n
        areas[cells.entity_id(c)] = length


@tack.kernel
def _polygon_normals(cells, out, areas, marks, hidden):
    for c in cells:
        origin = cells.side_position(c, 0, 0)
        n = tack.vec3(0.0, 0.0, 0.0)
        for k in range(cells.num_faces(c)):
            n += (cells.side_position(c, k, 0) - origin).cross(
                cells.side_position(c, k, 1) - origin)
        length = n.norm()
        if hidden == 1 and (tack.i32(marks[cells.entity_id(c)]) & 40) != 0:
            n = n - n
            length = length - length
        out[cells.entity_id(c)] = n / length if length > 0.0 else n
        areas[cells.entity_id(c)] = length


@tack.kernel
def _corner_normals(cells, normals, areas, weighted, start, out):
    # Each corner's cell normal, times the cell's area with weighted.
    for c in cells:
        first = start + cells.index(c) * cells.NUM_POINTS
        n = normals[cells.entity_id(c)]
        if weighted == 1:
            n = n * areas[cells.entity_id(c)]
        for j in range(cells.NUM_POINTS):
            out[first + j] = n


@tack.kernel
def _loop_corner_normals(cells, normals, areas, weighted, start, corners, out):
    # A polygon beside shapes: its normal at each of its points, in `corner_layout`'s
    # order (`polyhedral_corners`' offsets).
    for c in cells:
        first = start + corners[cells.index(c)]
        n = normals[cells.entity_id(c)]
        if weighted == 1:
            n = n * areas[cells.entity_id(c)]
        for j in range(cells.num_points(c)):
            out[first + j] = n


@tack.kernel
def _weighted(corners, owners, areas):
    for k in range(corners.shape[0]):
        corners[k] = corners[k] * areas[owners[k]]


@tack.kernel
def _entry_cells(offsets, owners):
    for c in range(offsets.shape[0] - 1):
        for k in range(offsets[c], offsets[c + 1]):
            owners[k] = c


@tack.kernel
def _unit(values):
    for i in range(values.shape[0]):
        length = values[i].norm()
        if length > 0.0:
            values[i] = values[i] / length


def _consistent(data):
    """``data`` with every connected region's polygons wound as its first cell is:
    union-find over the edges with each cell's parity to its root."""
    topology = data.topology
    faces = topology.faces()
    nf, n = faces.num_faces, data.num_cells
    pdt = tack.i64 if 2 * n + 1 > 2 ** 31 - 1 else tack.i32
    pairs = tack.empty((nf,), tack.vec(2, pdt))
    pairs.fill(-1)
    relations = tack.zeros((nf,), tack.i32)
    for group in data.launch_groups("cells", []):
        if group.count and group.shape.DIMENSION == 2:      # lines and vertices: no edges
            if group.shape.REFERENCE:
                _edge_relations(data.domain_view("cells", group, incidence="faces"),
                                faces.sides, pairs, relations, *_masks(data))
            else:                                           # polygons beside shapes
                _loop_relations(group.view(), faces.sides, pairs, relations, *_masks(data))
    parents = tack.array(2 * np.arange(n, dtype=pdt.numpy_dtype), pdt)
    changed = tack.zeros((1,), tack.i32)
    while nf:
        changed.fill(0)
        _hook(pairs, relations, parents, changed)
        _jump(parents)
        if not int(changed.numpy()[0]):
            break
    bad = tack.zeros((1,), tack.i32)
    if nf:
        _contradictions(pairs, relations, parents, bad)
    if bad[0]:
        raise ValueError(f"the surface is not orientable: {bad[0]} edges cannot be wound "
                         "consistently (a Moebius strip, or polygons folded through an edge)")
    offsets = arrays.materialize(topology.offsets)
    connectivity = tack.empty((arrays.size_of(topology.connectivity),),
                              arrays.dtype_of(topology.connectivity))
    tack.copyto(connectivity, arrays.materialize(topology.connectivity))
    for group in data.launch_groups("cells", []):
        if group.count and group.shape in _REVERSED:
            _rewound(data.domain_view("cells", group), parents, offsets,
                     _table(("reversed", group.shape), _REVERSED[group.shape]), connectivity)
        elif group.count and not group.shape.REFERENCE:
            _rewound_loops(group.view(), parents, offsets, connectivity)
    out = UnstructuredTopology(topology.types, topology.offsets, connectivity,
                               num_points=data.num_points, id_dtype=topology.id_dtype)
    return carry(data, out, points=Same(), cells=Same())


@tack.kernel
def _corner_cells(cells, start, out):
    # Each corner entry's cell, in point_links' layout (by cell and corner).
    for c in cells:
        first = start + cells.index(c) * cells.NUM_POINTS
        for j in range(cells.NUM_POINTS):
            out[first + j] = cells.entity_id(c)


@tack.kernel
def _loop_corner_cells(cells, start, corners, out):
    # A polygon beside shapes: its cell at each of its points, where
    # _loop_corner_normals puts its normal.
    for c in cells:
        first = start + corners[cells.index(c)]
        for j in range(cells.num_points(c)):
            out[first + j] = cells.entity_id(c)


@tack.kernel
def _link_flags(links, entry_cells, marks, flags):
    for i in range(links.shape[0]):
        flags[i] = 0 if (tack.i32(marks[entry_cells[links[i]]]) & 40) != 0 else 1


@tack.kernel
def _kept_links(links, flags, slots, out):
    for i in range(links.shape[0]):
        if flags[i] == 1:
            out[slots[i]] = links[i]


@tack.kernel
def _kept_offsets(offsets, slots, total, kept, out):
    for p in range(out.shape[0]):
        k = offsets[p]
        out[p] = slots[k] if k < total else kept


def _links_left(offsets, links, entry_cells, marks):
    """``(offsets, links)`` without the entries of hidden or refined cells, each
    point's others in their order."""
    total = links.shape[0]
    flags = tack.empty((total,), tack.i32)
    _link_flags(links, entry_cells, marks, flags)
    slots = tack.empty((total,), offsets.dtype)
    kept = exclusive_scan(flags, slots, total) if total else 0
    out_links = tack.empty((kept,), links.dtype)
    if kept:
        _kept_links(links, flags, slots, out_links)
    out_offsets = tack.empty((offsets.shape[0],), offsets.dtype)
    _kept_offsets(offsets, slots, total, kept, out_offsets)
    return out_offsets, out_links


@on_inputs_device
def surface_normals(data, consistent=True, weights=None):
    """A surface -- triangles, quads, pixels or polygons -- with its normals: the
    unit normal of each cell (Newell's, round its corners) as the cell field
    ``cell_normals``, and at each point the cells' around it summed and made unit
    as the point field ``normals`` (zero at a point of no cells), as VTK's
    ``vtkPolyDataNormals`` computes them without splitting. With ``weights="area"``
    each cell's normal is weighted by its area in the sum, so slivers -- which
    cut surfaces have where a crossing lands near a corner, and whose normals
    are far off the surface's -- count for little.

    With ``consistent``, the polygons are first wound alike: each connected
    region's as its first (lowest-numbered) cell winds, by union-find over the
    edges on the device, the cells that wind against it turned round (their
    corners reversed from the first), so their normals agree -- VTK's
    consistency, which is on by default there too. A surface no winding makes
    consistent (a Moebius strip) is refused, as is an edge of three or more
    polygons (the mesh's faces refuse it). A polygonal topology is consistent
    already (it refuses to be made otherwise), so it is left as it is. Lines
    and vertices among the polygons are kept as they are, with a zero normal
    that adds nothing at their points. The result keeps every field; the
    normals are dimensionless.

    Hidden and refined cells stay, marked, as the output is on the input's cells:
    each has a zero normal, adds nothing at its points, joins no region for the
    winding, and keeps its own."""
    if weights not in (None, "area"):
        raise ValueError(f"weights is None or 'area', not {weights!r}")
    if data.num_cells and data.topology.dimension != 2:
        raise ValueError(f"surface_normals takes 2D cells (a surface), not a mesh of "
                         f"dimension {data.topology.dimension}: extract_surface a solid "
                         "first")
    if _masks(data)[1]:
        # The marks change the answer, and the geometry's cache does not know them.
        topology, points, cell_normals = _normals(data, consistent, weights)
    else:
        topology, points, cell_normals = mesh_kept(
            data, ("surface_normals", bool(consistent), weights),
            lambda: _normals(data, consistent, weights))
    data = carry(data, topology, points=Same(), cells=Same())          # not the input
    data.fields["normals"] = Field(Values(data, "points"), points, "dimensionless")
    data.fields["cell_normals"] = Field(Values(data, "cells"), cell_normals, "dimensionless")
    return data


def _normals(data, consistent, weights):
    """``(topology, point normals, cell normals)``: the surface's topology, rewound
    where ``consistent`` winds it, and its normals' arrays."""
    if consistent and isinstance(data.topology, UnstructuredTopology) and data.num_cells:
        data = _consistent(data)
    n = data.num_cells
    cell_normals = _vectors(n, data.dtype)
    areas = tack.empty((n,), data.dtype)
    cell_normals.fill(0)                           # lines and vertices: none,
    areas.fill(0)                                  # nor any area to weigh by
    for group in data.launch_groups("cells", []):
        if not group.count or group.shape.DIMENSION != 2:
            continue
        view = data.domain_view("cells", group)
        if group.shape in _ROUND:
            _shape_normals(view, _table(("round", group.shape), _ROUND[group.shape]),
                           cell_normals, areas, *_masks(data))
        else:
            _polygon_normals(view, cell_normals, areas, *_masks(data))
    points = tack.empty((data.num_points,), tack.vec(3, data.dtype))
    points.fill(0)
    # Each cell's normal at each of its corners, as point_links lays them out
    # (by cell and corner on shapes, by cell_points' entries elsewhere).
    marks, hidden = _masks(data)
    entry_cells = None
    if getattr(data.topology, "reference_cells", True) or data.topology.mixed:
        start_of, total = corner_layout(data.topology)
        corners = _vectors(total, data.dtype)
        if hidden:
            entry_cells = tack.empty((total,), data.id_dtype)
        for group in data.launch_groups("cells", []):
            if not group.count:
                continue
            if group.shape.REFERENCE:
                _corner_normals(group.view(), cell_normals, areas, int(weights == "area"),
                                start_of[group.root], corners)
                if hidden:
                    _corner_cells(group.view(), start_of[group.root], entry_cells)
            else:                                           # polygons beside shapes
                loop_corners = polyhedral_corners(data.topology)[0]
                _loop_corner_normals(group.view(), cell_normals, areas,
                                     int(weights == "area"), start_of[group.root],
                                     loop_corners, corners)
                if hidden:
                    _loop_corner_cells(group.view(), start_of[group.root], loop_corners,
                                       entry_cells)
    else:
        point_offsets, point_ids = data.topology.cell_points()
        total = point_ids.shape[0]
        owners = tack.empty((total,), data.id_dtype)
        if total:
            _entry_cells(point_offsets, owners)
        corners = _take(cell_normals, owners)
        if weights == "area" and total:
            _weighted(corners, owners, areas)
        entry_cells = owners if hidden else None
    if total:
        offsets, links = data.topology.point_links()
        if hidden:
            # Over the corners of the cells left alone, in their order: the mean
            # the cells left once extracted give, to the bit.
            offsets, links = _links_left(offsets, links, entry_cells, marks)
        _average_links(offsets, links, corners, points)
        _unit(points)
    return data.topology, points, cell_normals
