# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""``contour`` of a structured grid by flying edges (issue #316), as VTK's
``vtkFlyingEdges3D`` and Viskores' flying edges: crossings found on the grid's
edges, counted along each row of points, numbered by a scan and written once,
so triangles share their points with no per-corner arrays and no sort.

Its triangles, points and the values carried are the general path's
(``contour._contour_points`` and its merge), up to the order of points and
triangles: every crossing is read through the same views (an image's and a
rectilinear grid's positions by ``get_ijk``, as its cells read them), placed
from the end first by position, and named by its end where it lands on one
(``_end_hit``); every cell takes the same triangles of the hexahedron's case
table; a triangle two of whose points are one point is dropped.

Each point owns the crossings of its edges along +x, +y and +z, and the point
itself where a crossing lands on it; per value, its edges' codes (none, between
the ends, at this end, at the far end; two bits an edge) are found once, then
each row of points counts what it emits and each row of cells its triangles,
in the order (row, value, along x). The triangles' pass walks each row of
cells along x with a running count for each of the four rows of points around
it, as VTK's does, so a crossing's number is found without a table a point.
"""

import tack
from tack import arrays
from tack.algorithms.scan import exclusive_scan
from tack.dataset import ids, shapes
from tack.dataset.algorithms._common import _crossing, _end_hit, _first_end

# An edge's code, two bits an axis in a point's flag (bits 2a, 2a + 1).
_NONE = tack.constant(0, tack.i32)
_BETWEEN = tack.constant(1, tack.i32)
_AT_START = tack.constant(2, tack.i32)
_AT_END = tack.constant(3, tack.i32)


@tack.func
def _corner_of(cells, i, j, k):
    """A cell holding point ``(i, j, k)`` and the point's corner in it, so the
    point is read through the cells' views, as the general path reads it."""
    ci = min(i, cells.cx - 1)
    cj = min(j, cells.cy - 1)
    ck = min(k, cells.cz - 1)
    a = i - ci
    b = j - cj
    d = k - ck
    return [ci, cj, ck], 4 * d + 2 * b + (a ^ b)


@tack.func
def _point_at(cells, u, i, j, k):
    """``(position, value)`` of point ``(i, j, k)``."""
    c, corner = _corner_of(cells, i, j, k)
    return cells.corner_position(c, corner), u.corner_value(c, corner)


@tack.func
def _position_at(cells, i, j, k):
    c, corner = _corner_of(cells, i, j, k)
    return cells.corner_position(c, corner)


@tack.func
def _crosses(v0, v1, level):
    # A cell with a NaN corner has no surface, so an edge to a NaN is in none of
    # the triangles.
    return v0 == v0 and v1 == v1 and (v0 >= level) != (v1 >= level)


@tack.kernel
def _edge_codes(values, levels, nlevels, nx, ny, nz, flags):
    # Each point's edges along +x, +y and +z, per value: two bits each, here
    # crossed or not (_BETWEEN); _landings finds those that land on an end.
    npoints = nx * ny * nz
    for k, j, i in tack.ndindex(nz, ny, nx):
        p = i + nx * (j + ny * k)
        v0 = values[p]
        vx = values[p + 1] if i + 1 < nx else v0
        vy = values[p + nx] if j + 1 < ny else v0
        vz = values[p + nx * ny] if k + 1 < nz else v0
        for v in range(nlevels):
            level = levels[v]
            f = 0
            if _crosses(v0, vx, level):
                f = 1
            if _crosses(v0, vy, level):
                f |= 4
            if _crosses(v0, vz, level):
                f |= 16
            if v == 0 and v0 != v0:
                f |= _MISSING
            flags[v * npoints + p] = tack.u8(f)


@tack.kernel
def _landings(cells, values, levels, nlevels, nx, ny, nz, flags):
    # Each crossed edge whose crossing lands on an end (_end_hit), recoded
    # _AT_START or _AT_END; positions, which an image computes, are read through
    # the cells' views as the general path reads them, and only here. One
    # (row of points, value) a thread, each writing only its own points' flags.
    nrows = ny * nz
    npoints = nx * ny * nz
    for q in range(nrows * nlevels):
        r = q // nlevels
        v = q - r * nlevels
        j = r % ny
        k = r // ny
        level = levels[v]
        for i in range(nx):
            p = i + nx * r
            base = v * npoints + p
            f = tack.i32(flags[base])
            if f != 0:
                for axis in range(3):
                    if ((f >> (2 * axis)) & 3) == _BETWEEN:
                        ii = i + (1 if axis == 0 else 0)
                        jj = j + (1 if axis == 1 else 0)
                        kk = k + (1 if axis == 2 else 0)
                        b = ii + nx * (jj + ny * kk)
                        hit = _end_hit(level, p, b, values[p], values[b],
                                       _position_at(cells, i, j, k),
                                       _position_at(cells, ii, jj, kk))
                        if hit == p:
                            f += (_AT_START - _BETWEEN) << (2 * axis)
                        elif hit == b:
                            f += (_AT_END - _BETWEEN) << (2 * axis)
                flags[base] = tack.u8(f)


@tack.func
def _code(flags, at, axis):
    return (tack.i32(flags[at]) >> (2 * axis)) & 3


@tack.func
def _is_hit(flags, base, i, j, k, nx, ny):
    """Whether a crossing lands on point ``(i, j, k)`` (``base`` its flag's index):
    one of its own edges' at its start, or a lower neighbour's at its end."""
    hit = 0
    for axis in range(3):
        if _code(flags, base, axis) == _AT_START:
            hit = 1
    if i > 0 and _code(flags, base - 1, 0) == _AT_END:
        hit = 1
    if j > 0 and _code(flags, base - nx, 1) == _AT_END:
        hit = 1
    if k > 0 and _code(flags, base - nx * ny, 2) == _AT_END:
        hit = 1
    return hit


#: A point's flag bit saying a crossing lands on it (``_point_counts`` sets it).
_LANDED = tack.constant(64, tack.i32)
#: The first value's flag bit saying the point's value is NaN.
_MISSING = tack.constant(128, tack.i32)


@tack.func
def _landed(flags, base):
    return 1 if (tack.i32(flags[base]) & _LANDED) != 0 else 0


@tack.func
def _emitted(flags, base):
    """How many points a point emits (its flag at ``base``, with ``_LANDED`` set):
    itself if a crossing lands on it, then each of its edges crossed between its
    ends."""
    f = tack.i32(flags[base])
    count = 1 if (f & _LANDED) != 0 else 0
    for axis in range(3):
        if ((f >> (2 * axis)) & 3) == _BETWEEN:
            count += 1
    return count


@tack.kernel
def _point_counts(flags, nlevels, nx, ny, nz, counts, nans):
    # Each (row of points, value)'s points, and each row's NaNs, which can leave a
    # crossing no cell uses.
    nrows = ny * nz
    npoints = nx * ny * nz
    for q in range(nrows * nlevels):
        r = q // nlevels
        v = q - r * nlevels
        j = r % ny
        k = r // ny
        total = 0
        missing = 0
        for i in range(nx):
            # Marked here, by the one thread that walks this row for this value;
            # others read only the bits of the point's own edges.
            base = v * npoints + i + nx * r
            if _is_hit(flags, base, i, j, k, nx, ny) == 1:
                flags[base] = flags[base] | tack.u8(_LANDED)
            total += _emitted(flags, base)
            if v == 0 and (tack.i32(flags[base]) & _MISSING) != 0:
                missing += 1
        counts[q] = total
        if v == 0:
            nans[r] = missing


@tack.func
def _edge_of(hexa, e, ci, cj, ck):
    """Local edge ``e`` of cell ``(ci, cj, ck)``: its axis and its lower point's
    ``(i, j, k)``."""
    ja = hexa.edge_corner(e, 0)
    jb = hexa.edge_corner(e, 1)
    ab = (ja >> 1) & 1
    bb = (jb >> 1) & 1
    a0 = (ja & 1) ^ ab
    a1 = (jb & 1) ^ bb
    d0 = ja >> 2
    d1 = jb >> 2
    axis = 0 if a0 != a1 else (1 if ab != bb else 2)
    return axis, ci + min(a0, a1), cj + min(ab, bb), ck + min(d0, d1)


@tack.func
def _landing(flags, v, npoints, hexa, e, ci, cj, ck, nx, ny):
    """The point a crossing of local edge ``e`` lands on (its global id), or -1
    for one between the edge's ends."""
    axis, i, j, k = _edge_of(hexa, e, ci, cj, ck)
    p = i + nx * (j + ny * k)
    code = _code(flags, v * npoints + p, axis)
    out = tack.i64(-1)
    if code == _AT_START:
        out = p
    elif code == _AT_END:
        out = p + (1 if axis == 0 else (nx if axis == 1 else nx * ny))
    return out


@tack.func
def _kept(flags, v, npoints, hexa, case, t, ci, cj, ck, nx, ny):
    """Whether triangle ``t`` of the case keeps three distinct points (two crossings
    landing on one point leave none)."""
    a = _landing(flags, v, npoints, hexa, hexa.contour_edge(case, 3 * t), ci, cj, ck, nx, ny)
    b = _landing(flags, v, npoints, hexa, hexa.contour_edge(case, 3 * t + 1), ci, cj, ck,
                 nx, ny)
    c = _landing(flags, v, npoints, hexa, hexa.contour_edge(case, 3 * t + 2), ci, cj, ck,
                 nx, ny)
    return not ((a >= 0 and (a == b or a == c)) or (b >= 0 and b == c))


@tack.func
def _cell_case(cells, u, c, level):
    # As contour._cell_case: the corners at or above the level, as bits; none
    # where a corner is NaN.
    case = 0
    nan = 0
    for j in range(cells.NUM_POINTS):
        x = u.corner_value(c, j)
        if x >= level:
            case |= 1 << j
        if x != x:
            nan = 1
    return 0 if nan == 1 else case


@tack.func
def _crossed(point_counts, nlevels, v, cj, ck, ny):
    """Whether any of the four rows of points around row of cells ``(cj, ck)``
    emits a point for value ``v``: if none does, none of its cells' edges is
    crossed, and it has no triangles."""
    r0 = cj + ny * ck
    return (point_counts[r0 * nlevels + v] + point_counts[(r0 + 1) * nlevels + v]
            + point_counts[(r0 + ny) * nlevels + v]
            + point_counts[(r0 + ny + 1) * nlevels + v]) > 0


@tack.kernel
def _triangle_counts(cells, u, hexa, flags, point_counts, levels, nlevels, nx, ny, nz,
                     counts):
    # Each (row of cells, value)'s triangles.
    cx = nx - 1
    cy = ny - 1
    npoints = nx * ny * nz
    for q in range(cy * (nz - 1) * nlevels):
        r = q // nlevels
        v = q - r * nlevels
        cj = r % cy
        ck = r // cy
        level = levels[v]
        total = 0
        crossed = _crossed(point_counts, nlevels, v, cj, ck, ny)
        for ci in range(cx if crossed else 0):
            case = _cell_case(cells, u, [ci, cj, ck], level)
            for t in range(hexa.contour_count(case)):
                if _kept(flags, v, npoints, hexa, case, t, ci, cj, ck, nx, ny):
                    total += 1
        counts[q] = total


@tack.kernel
def _points(cells, u, flags, levels, nlevels, nx, ny, nz, starts, total, positions, edges,
            weights):
    # Each (row of points, value)'s points, in its order: a point a crossing lands
    # on, then its edges' crossings between their ends along x, y and z.
    nrows = ny * nz
    npoints = nx * ny * nz
    for q in range(nrows * nlevels):
        r = q // nlevels
        v = q - r * nlevels
        j = r % ny
        k = r // ny
        level = levels[v]
        at = starts[q]
        end = starts[q + 1] if q + 1 < nrows * nlevels else total
        for i in range(nx if end > at else 0):
            p = i + nx * r
            base = v * npoints + p
            x0, v0 = _point_at(cells, u, i, j, k)
            if _landed(flags, base) == 1:
                positions[at] = x0
                edges[at] = [p, p]
                weights[at] = 0.0
                at += 1
            for axis in range(3):
                if _code(flags, base, axis) == _BETWEEN:
                    ii = i + (1 if axis == 0 else 0)
                    jj = j + (1 if axis == 1 else 0)
                    kk = k + (1 if axis == 2 else 0)
                    b = ii + nx * (jj + ny * kk)
                    x1, v1 = _point_at(cells, u, ii, jj, kk)
                    a = p
                    xa = x0
                    xb = x1
                    va = v0
                    vb = v1
                    # From the end first by position, as the general path.
                    if not _first_end(x0, x1, p, b):
                        a = b
                        b = p
                        xa = x1
                        xb = x0
                        va = v1
                        vb = v0
                    w = _crossing(level, va, vb)
                    positions[at] = xa + w * (xb - xa)
                    edges[at] = [a, b]
                    weights[at] = w
                    at += 1


@tack.func
def _pick(s, a, b, c, d):
    out = a
    if s == 1:
        out = b
    elif s == 2:
        out = c
    elif s == 3:
        out = d
    return out


@tack.func
def _slot(flags, base, axis, code, hit, start, far_start):
    """The number of a crossing of the edge along ``axis`` of a point whose first
    number is ``start`` (``hit`` 1 if a crossing lands on it, which comes first):
    its own between its ends, the point itself, or the far point (whose first
    number is ``far_start``, its own landing)."""
    out = start
    if code == _BETWEEN:
        out = start + hit
        for a in range(3):
            if a < axis and _code(flags, base, a) == _BETWEEN:
                out += 1
    elif code == _AT_END:
        out = far_start
    return out


@tack.kernel
def _triangles(cells, u, hexa, flags, levels, nlevels, nx, ny, nz, point_starts,
               triangle_starts, triangles, rows, sources, which):
    # Each (row of cells, value)'s triangles, walking along x with the next
    # number of each of the four rows of points around it.
    cx = nx - 1
    cy = ny - 1
    npoints = nx * ny * nz
    for q in range(cy * (nz - 1) * nlevels):
        r = q // nlevels
        v = q - r * nlevels
        cj = r % cy
        ck = r // cy
        level = levels[v]
        at = triangle_starts[q]
        end = triangle_starts[q + 1] if q + 1 < cy * (nz - 1) * nlevels else triangles
        # Rows (cj, ck), (cj + 1, ck), (cj, ck + 1), (cj + 1, ck + 1): the first
        # number of each one's point at ci.
        r0 = cj + ny * ck
        r1 = r0 + 1
        r2 = r0 + ny
        r3 = r2 + 1
        n0 = point_starts[r0 * nlevels + v]
        n1 = point_starts[r1 * nlevels + v]
        n2 = point_starts[r2 * nlevels + v]
        n3 = point_starts[r3 * nlevels + v]
        for ci in range(cx if end > at else 0):
            # The next point's first number in each row.
            m0 = n0 + _emitted(flags, v * npoints + ci + nx * r0)
            m1 = n1 + _emitted(flags, v * npoints + ci + nx * r1)
            m2 = n2 + _emitted(flags, v * npoints + ci + nx * r2)
            m3 = n3 + _emitted(flags, v * npoints + ci + nx * r3)
            case = _cell_case(cells, u, [ci, cj, ck], level)
            cell = ci + cx * (cj + cy * ck)
            for t in range(hexa.contour_count(case)):
                if _kept(flags, v, npoints, hexa, case, t, ci, cj, ck, nx, ny):
                    for corner in range(3):
                        e = hexa.contour_edge(case, 3 * t + corner)
                        axis, i, j, k = _edge_of(hexa, e, ci, cj, ck)
                        s = (j - cj) + 2 * (k - ck)
                        start = _pick(s, n0, n1, n2, n3) if i == ci else _pick(s, m0, m1, m2,
                                                                                 m3)
                        base = v * npoints + i + nx * (j + ny * k)
                        code = _code(flags, base, axis)
                        # The far point: along x the next point of the row; along y
                        # or z the row above at the same i.
                        far = _pick(s + (2 if axis == 2 else 1), n0, n1, n2, n3)
                        if axis == 0:
                            far = _pick(s, m0, m1, m2, m3)
                        elif i != ci:
                            far = _pick(s + (2 if axis == 2 else 1), m0, m1, m2, m3)
                        hit = _landed(flags, base)
                        rows[3 * at + corner] = _slot(flags, base, axis, code, hit, start, far)
                    sources[at] = cell
                    which[at] = v
                    at += 1
            n0 = m0
            n1 = m1
            n2 = m2
            n3 = m3


@tack.kernel
def _mark_used(rows, used):
    for t in range(rows.shape[0]):
        used[rows[t]] = 1


@tack.kernel
def _renumbered(rows, new_ids):
    for t in range(rows.shape[0]):
        rows[t] = new_ids[rows[t]]


@tack.kernel
def _taken(kept, positions, edges, weights, out_positions, out_edges, out_weights):
    for n in range(kept.shape[0]):
        out_positions[n] = positions[kept[n]]
        out_edges[n] = edges[kept[n]]
        out_weights[n] = weights[kept[n]]


def serves(data, field):
    """Whether ``data``'s contour of ``field`` can go by flying edges: a structured
    grid of solid cells with an H1 geometry and a point field, both of order 1."""
    from tack.dataset.spaces import H1
    from tack.dataset.topology import StructuredTopology

    dims = getattr(data.topology, "point_dims", ())
    return (isinstance(data.topology, StructuredTopology) and len(dims) == 3
            and min(dims) >= 2 and isinstance(data.geometry.space, H1)
            and data.geometry.space.order == 1 and isinstance(field.space, H1)
            and field.space.order == 1)


def surface(data, field, levels, nlevels):
    """``(points, edges, weights, rows, sources, which)``: the contour's merged points
    with the edges and weights they interpolate along, its triangles' rows of
    points, their cells and their values' indices (see the module's docstring)."""
    from tack.dataset.algorithms._common import _compact

    nx, ny, nz = (int(d) for d in data.topology.point_dims)
    npoints, nrows = nx * ny * nz, ny * nz
    ncrows = (ny - 1) * (nz - 1)
    group = next(g for g in data.launch_groups("cells", [field]) if g.count)
    cells = data.domain_view("cells", group)
    u = field.view(group)
    hexa = shapes.Hexahedron()
    idt = data.id_dtype
    odt = ids.for_output(data, max(3 * nlevels * (npoints + data.num_cells * 5), 1))
    flags = tack.empty((nlevels * npoints,), tack.u8)
    _edge_codes(field.values, levels, nlevels, nx, ny, nz, flags)
    _landings(cells, field.values, levels, nlevels, nx, ny, nz, flags)
    point_counts = tack.empty((nrows * nlevels,), odt)
    nans = tack.empty((nrows,), tack.i64)
    _point_counts(flags, nlevels, nx, ny, nz, point_counts, nans)
    triangle_counts = tack.empty((ncrows * nlevels,), odt)
    _triangle_counts(cells, u, hexa, flags, point_counts, levels, nlevels, nx, ny, nz,
                     triangle_counts)
    count = exclusive_scan(point_counts, point_counts, nrows * nlevels)
    triangles = exclusive_scan(triangle_counts, triangle_counts, ncrows * nlevels)
    positions = tack.empty((count,), tack.vec(3, data.dtype))
    edges = tack.empty((count,), tack.vec(2, idt))
    weights = tack.empty((count,), arrays.dtype_of(field.values))
    if count:
        _points(cells, u, flags, levels, nlevels, nx, ny, nz, point_counts, count, positions,
                edges, weights)
    rows = tack.empty((3 * triangles,), odt)
    sources = tack.empty((triangles,), idt)
    which = tack.empty((triangles,), tack.i32)
    if triangles:
        _triangles(cells, u, hexa, flags, levels, nlevels, nx, ny, nz, point_counts,
                   triangle_counts, triangles, rows, sources, which)
    del flags, point_counts, triangle_counts
    if count and int(nans.sum()):
        # A crossing whose cells all have a NaN corner is in no triangle: kept out,
        # as the general path never makes it.
        used = tack.zeros((count,), tack.i32)
        if triangles:
            _mark_used(rows, used)
        new_ids, kept, kept_count = _compact(used, odt)
        if kept_count < count:
            if triangles:
                _renumbered(rows, new_ids)
            out = (tack.empty((kept_count,), positions.dtype),
                   tack.empty((kept_count,), edges.dtype), tack.empty((kept_count,), weights.dtype))
            if kept_count:
                _taken(kept, positions, edges, weights, *out)
            positions, edges, weights = out
            count = kept_count
    return positions, edges, weights, rows, sources, which, count, triangles, odt


__all__ = []
