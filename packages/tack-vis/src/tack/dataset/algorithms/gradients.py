# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""``gradients``: the gradient of ``field`` -- ``H1``, ``L2`` or ``Constant``, scalar or vector -- at each cell's parametric center (``at="cells"``) or at the points (``at="points"``: each cell's gradient at the point, averaged over the cells around it, as VTK's vtkGradientFilter and Viskores' point gradients do). On polyhedra, which have no reference element, each cell's gradient is Green–Gauss over its faces fanned from their points' means."""

import tack
from tack import arrays
from tack.arrays import materialize, size_of, width_of
from tack.dataset import shapes, units
from tack.dataset.algorithms._common import _field, _vectors, _with_basis
from tack.dataset.algorithms._links import _average_links
from tack.dataset.carry import _take
from tack.dataset.dataset import Field, for_each
from tack.dataset.spaces import H1, L2, Constant, Values
from tack.dataset.topology import corner_layout
from tack.runtime.dispatch import on_inputs_device

__all__ = ['gradients', 'world_gradient', 'world_hessian', 'world_jet']


@tack.func
def world_gradient(cells, u, c, pc):
    """`u`'s gradient in the world at parametric `pc` of cell `c`, as its view gives it:
    from its basis by default (`_BasisGradient`), the gradient it was given
    (`Field(..., gradient=)`), or a space's own. For kernels that need a field's
    gradient where `gradients` does not put it."""
    return u.world_gradient(cells, c, pc)


@tack.func
def world_hessian(cells, u, c, pc):
    """`u`'s second derivatives in the world at parametric `pc` of solid cell `c`, a
    symmetric 3x3 matrix, as its view gives them (`world_jet`). From the basis: by the
    chain rule through the geometry, the parametric Hessian less the geometry's
    curvature along the world gradient, taken to world axes on both sides by the
    inverse Jacobian; analytic on every shape at orders 1 and 2, and at any order on
    tensor shapes, simplices and wedges; zero on surfaces, lines and vertices. NaN for
    a field given only its gradient."""
    _, _, h = world_jet(cells, u, c, pc)
    return h


@tack.func
def world_jet(cells, u, c, pc):
    """Scalar `u`'s value, world gradient and world Hessian at parametric `pc` of cell
    `c`, as its view gives them, from one pass over the cell's values where it can:
    what a kernel needing several of them reads once. The gradient is
    `world_gradient`'s, and the Hessian `world_hessian`'s."""
    return u.world_jet(cells, c, pc)


@tack.func
def _in_plane(m, d):
    """The world vector in the plane of `m`'s first two columns whose dots with them
    are `d[0]` and `d[1]`."""
    a = tack.vec3(m[0, 0], m[1, 0], m[2, 0])
    b = tack.vec3(m[0, 1], m[1, 1], m[2, 1])
    aa = a.dot(a)
    ab = a.dot(b)
    bb = b.dot(b)
    det = aa * bb - ab * ab
    return ((bb * d[0] - ab * d[1]) * a + (aa * d[1] - ab * d[0]) * b) / det


class _BasisGradient:
    """A field view's world gradient from its basis: on every field view with a
    ``parametric_gradient`` and no other source (``_derivative_mixins``)."""

    @tack.func
    def world_gradient(self, cells, c, pc):
        # d(u)/d(x) from d(u)/d(pc), through the geometry's Jacobian J. A solid's is
        # J^-T d(u)/d(pc). A 2D cell's lies in its tangent plane, spanned by J's
        # columns a and b: g = alpha a + beta b with g . a and g . b the parametric
        # derivatives, VTK's and Viskores' in-plane gradient (no area: NaN, as a
        # solid of no volume). Lines and vertices: zero.
        g = tack.vec3(0.0, 0.0, 0.0)
        if cells.DIMENSION == 3:
            g = cells.geometry_jacobian(c, pc).inverse().transpose() @ self.parametric_gradient(c, pc)
        elif cells.DIMENSION == 2:
            g = _in_plane(cells.geometry_jacobian(c, pc), self.parametric_gradient(c, pc))
        return g


class _BasisJet:
    """A field view's value, world gradient and world Hessian from its basis, in one
    pass: on every field view with a ``parametric_jet``."""

    @tack.func
    def world_jet(self, cells, c, pc):
        v, pg, ph = self.parametric_jet(c, pc)
        g = tack.vec3(0.0, 0.0, 0.0)
        h = tack.mat3([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        if cells.DIMENSION == 3:
            inverse = cells.geometry_jacobian(c, pc).inverse()
            g = inverse.transpose() @ pg
            h = inverse.transpose() @ (ph - cells.geometry_curvature(c, pc, g)) @ inverse
        elif cells.DIMENSION == 2:
            g = _in_plane(cells.geometry_jacobian(c, pc), pg)
        return v, g, h


class _GivenGradient:
    """A field given its gradient (``Field(..., gradient=)``): ``world_gradient`` is that
    field's value at ``pc`` of cell ``c``, and ``world_jet`` pairs it with the field's
    value and a Hessian of NaN, which a gradient alone does not determine."""

    @tack.func
    def world_gradient(self, cells, c, pc):
        return self.given_gradient.value(c, pc)

    @tack.func
    def world_jet(self, cells, c, pc):
        n = self.no_hessian
        return (self.value(c, pc), self.given_gradient.value(c, pc),
                tack.mat3([n, n, n], [n, n, n], [n, n, n]))


@tack.kernel
def _gradients(cells, u, out):
    for c in cells:
        out[cells.entity_id(c)] = world_gradient(cells, u, c, cells.parametric_center())


@tack.kernel
def _node_gradients(cells, u, out):
    # The gradient at each of the cell's nodes of the target space.
    for c in cells:
        for j in range(cells.node_count(c)):
            out[cells.node_dof(c, j)] = world_gradient(cells, u, c, cells.node_parametric_point(c, j))


#: A corner whose scaled Jacobian (the determinant over its columns' lengths) is
#: within this of 0 is flat, two of the cell's faces meeting there in one plane,
#: whatever rounding left of the 0 -- in f32, positions of size 6 round a cell of
#: size 1e-3 by 4e-4 of it, and AhmedML's flat corners came out at 1e-4 to 2e-4 --
#: its angle under a tenth of a degree; below its negative, the cell is tangled.
#: The same band applies to f64 data, where rounding is far smaller: a corner
#: within it is as good as flat, its gradient a thousand times the cell's and more.
_FLAT = tack.constant(1e-3)

#: A cell's Green-Gauss gradient is trusted while its signed volume is more than
#: this of the sum of its sub-tetrahedra's absolute volumes (1 for a cell that
#: does not fold). A folded cell's parts cancel as it turns inside out, and its
#: gradient grows as one over the fraction: on a hexahedron folded by one point
#: (`test_gradients._tangled`), the largest point gradient was 7 at 0.11, 21 at
#: 0.04 and 612 at 0.002.
_THIN = tack.constant(0.1)


@tack.func
def _scaled(m):
    """A solid's Jacobian ``m``, scaled: its determinant over the product of its
    columns' lengths (at a hexahedron's corner, its three edges there), 1 for a
    right angle, 0 flat, negative folded; 0 for a column of no length."""
    a = tack.vec3(m[0, 0], m[1, 0], m[2, 0])
    b = tack.vec3(m[0, 1], m[1, 1], m[2, 1])
    d = tack.vec3(m[0, 2], m[1, 2], m[2, 2])
    lengths = a.norm() * b.norm() * d.norm()
    scaled = lengths - lengths
    if lengths > 0.0:
        scaled = m.determinant() / lengths
    return scaled


@tack.func
def _corner_kind(cells, c, pc):
    """``(singular, folded)`` where the geometry's Jacobian at parametric `pc` of cell
    `c` is: singular 1 where it has no inverse -- a solid's flat (its scaled
    Jacobian within ``_FLAT`` of 0), a 2D cell's in-plane determinant
    (`_in_plane`'s) exactly 0 -- and folded 1 where a solid's is below ``-_FLAT``
    (a tetrahedron, whose Jacobian is one, never: inside out is ``orient``'s)."""
    m = cells.geometry_jacobian(c, pc)
    s = 0
    folded = 0
    if cells.DIMENSION == 3:
        scaled = _scaled(m)
        if abs(scaled) <= _FLAT:
            s = 1
        elif scaled < -_FLAT and cells.ID != shapes.TETRA:
            folded = 1
    elif cells.DIMENSION == 2:
        a = tack.vec3(m[0, 0], m[1, 0], m[2, 0])
        b = tack.vec3(m[0, 1], m[1, 1], m[2, 1])
        aa = a.dot(a)
        ab = a.dot(b)
        bb = b.dot(b)
        if aa * bb - ab * ab == 0.0:
            s = 1
    return s, folded


@tack.func
def _shape_green_gauss(cells, u, c):
    """A solid cell's gradient as a polyhedron's (``_polyhedral_gradients``): Green-Gauss
    over its faces from the shape's tables, each fanned from its points' mean, so a
    linear field's is exact, without the map's Jacobian. Positions and values from
    the cell's first corner. Faces wound all outward or all inward give the same
    gradient, the volume's sign cancelling. Also whether it is trusted: 1 where its
    signed volume is more than ``_THIN`` of its sub-tetrahedra's absolute volumes
    (each fan triangle's with the first corner)."""
    origin = cells.corner_position(c, 0)
    u0 = u.corner_value(c, 0)
    six = origin[0] - origin[0]
    whole = six
    total = origin - origin
    for k in range(cells.NUM_FACES):
        size = cells.face_num_points(k)
        first = cells.corner_position(c, cells.face_corner(k, 0)) - origin
        u_first = u.corner_value(c, cells.face_corner(k, 0)) - u0
        a = first
        ua = u_first
        points = first
        values = u_first
        newell = origin - origin
        spun = origin - origin
        along = origin - origin
        for j in range(1, size + 1):
            b = first
            ub = u_first
            if j < size:
                b = cells.corner_position(c, cells.face_corner(k, j)) - origin
                ub = u.corner_value(c, cells.face_corner(k, j)) - u0
                points += b
                values += ub
            n = a.cross(b)
            both = ua + ub
            newell += n
            spun += n * both
            along += (b - a) * both
            a = b
            ua = ub
        apex = points / size
        six += apex.dot(newell)
        total += spun + along.cross(apex) + newell * (values / size)
        # The sub-tetrahedra over this face, apex . (a x b) each, again.
        a = first
        for j in range(1, size + 1):
            b = first
            if j < size:
                b = cells.corner_position(c, cells.face_corner(k, j)) - origin
            whole += abs(apex.dot(a.cross(b)))
            a = b
    trusted = 1 if abs(six) > _THIN * whole else 0
    return total / six, trusted


@tack.kernel
def _corner_gradients(cells, u, start, out, singular):
    # singular, per corner: 0 a gradient; 1 none, left out of the point's mean; 2
    # left out too, but its cell's Green-Gauss gradient is in out, for a point all
    # of whose corners are left out (_without_singular).
    for c in cells:
        first = start + cells.index(c) * cells.NUM_POINTS
        # Tangled: the map folds over (a corner's scaled Jacobian below -_FLAT), so
        # its inverse, which the corner gradients take, is wrong near the fold and
        # unbounded at it. The cell's gradient is then its own as a polyhedron's,
        # at every corner, unless the cell is nearly as much inside out as not
        # (_THIN), where that grows without bound too: then a corner whose map does
        # not fold keeps its own, and the folded ones are left out.
        tangled = 0
        flats = 0
        for j in range(cells.NUM_POINTS):
            singular[first + j] = 0
            if cells.ID == shapes.PYRAMID and j == 4:
                # The Jacobian is singular at a pyramid's apex. Extrapolate from
                # two points just below it on the axis, as vtkPyramid does.
                ga = world_gradient(cells, u, c, tack.vec3(0.5, 0.5, 0.998))
                gb = world_gradient(cells, u, c, tack.vec3(0.5, 0.5, 0.996))
                out[first + j] = 2.0 * ga - gb
            else:
                pc = cells.parametric_point(j)
                flat, folded = _corner_kind(cells, c, pc)
                if folded == 1:
                    tangled = 1
                    singular[first + j] = 3          # folded: settled below
                if flat == 1:
                    # Singular at this corner too, where two of the cell's faces
                    # meet flat (a corner on a straight edge of a face), or all but
                    # by rounding (AhmedML's hexahedra, which gave gradients to
                    # 1e9): its gradient grows without bound toward the corner and
                    # has no value there, so the cell is left out of the point's
                    # average.
                    out[first + j] = tack.vec3(0.0, 0.0, 0.0)
                    singular[first + j] = 1
                    flats = 1
                else:
                    out[first + j] = world_gradient(cells, u, c, pc)
        if cells.DIMENSION == 3 and (tangled == 1 or flats == 1):
            g, trusted = _shape_green_gauss(cells, u, c)
            for j in range(cells.NUM_POINTS):
                kind = singular[first + j]
                if tangled == 1 and trusted == 1:
                    out[first + j] = g
                    singular[first + j] = 0
                elif kind == 3:
                    out[first + j] = tack.vec3(0.0, 0.0, 0.0)
                    singular[first + j] = 1
                elif kind == 1 and trusted == 1:
                    # A flat corner keeps the cell's Green-Gauss gradient, which is
                    # bounded there, for a point with no other gradient.
                    out[first + j] = g
                    singular[first + j] = 2


@tack.kernel
def _without_singular(offsets, entries, singular, corners, out, nan):
    # A point's mean over its corners that have a gradient (singular 0); where none
    # has, over those left out with their cell's Green-Gauss gradient (2); NaN where
    # there is neither.
    for p in range(out.shape[0]):
        left_out = 0
        for k in range(offsets[p], offsets[p + 1]):
            if singular[entries[k]] != 0:
                left_out += 1
        if left_out > 0:
            total = out[p] - out[p]          # in the gradients' own type: Metal has no f64
            kept = 0
            for k in range(offsets[p], offsets[p + 1]):
                if singular[entries[k]] == 0:
                    total += corners[entries[k]]
                    kept += 1
            if kept == 0:
                for k in range(offsets[p], offsets[p + 1]):
                    if singular[entries[k]] == 2:
                        total += corners[entries[k]]
                        kept += 1
            if kept == 0:
                out[p] = tack.vec3(nan, nan, nan)
            else:
                out[p] = total / kept


@tack.kernel
def _polyhedral_gradients(cells, u, out):
    for c in cells:
        # Green-Gauss: (1/V) times the sum over the cell's boundary of each triangle's
        # outward area vector times the mean of its three values, the faces walked
        # outward and fanned from their points' means -- in position and in value,
        # so a linear field's gradient is exact. The triangles are cell_geometry's,
        # so V is its volume. Positions are taken from the cell's first point and
        # values from its value there (the area vectors of a closed surface sum to
        # zero), so cells far from the origin keep their precision.
        origin = cells.side_position(c, 0, 0)
        u0 = u[cells.side_point_id(c, 0, 0)]
        six = 0.0                                  # six times the volume
        total = tack.vec3(0.0, 0.0, 0.0)           # 2 A_t times 3 mean(u) on each triangle
        for k in range(cells.num_faces(c)):
            # One walk round the face. With F its points' mean, triangle j's
            # 2 A_j = p_j x p_j+1 + (p_j+1 - p_j) x F, so the face's share is
            # S + W x F + u_F N, with S = sum (p_j x p_j+1)(u_j + u_j+1),
            # W = sum (p_j+1 - p_j)(u_j + u_j+1) and N = sum p_j x p_j+1 (the
            # Newell vector, sum 2 A_j); its share of 6V is F . N. F and u_F are
            # known only at the end, and none of the sums needs them.
            size = cells.side_num_points(c, k)
            first = cells.side_position(c, k, 0) - origin
            u_first = u[cells.side_point_id(c, k, 0)] - u0
            a = first
            ua = u_first
            points = first
            values = u_first
            newell = tack.vec3(0.0, 0.0, 0.0)
            spun = tack.vec3(0.0, 0.0, 0.0)
            along = tack.vec3(0.0, 0.0, 0.0)
            for j in range(1, size + 1):
                b = first
                ub = u_first
                if j < size:
                    b = cells.side_position(c, k, j) - origin
                    ub = u[cells.side_point_id(c, k, j)] - u0
                    points += b
                    values += ub
                n = a.cross(b)
                both = ua + ub
                newell += n
                spun += n * both
                along += (b - a) * both
                a = b
                ua = ub
            apex = points / size
            six += apex.dot(newell)
            total += spun + along.cross(apex) + newell * (values / size)
        # (1/V) sum A_t mean(u)_t = sum (2 A_t)(3 mean(u)_t) / 6V. No volume: NaN.
        out[cells.entity_id(c)] = total / six


@tack.kernel
def _polygon_gradients(cells, u, out):
    for c in cells:
        # In-plane Green-Gauss: (1/A) times the sum over the edges of each edge's
        # outward in-plane normal (its length long) times its mean value, with N
        # the Newell vector (2A along the normal): sum (u_k + u_k+1) ((p_k+1 - p_k)
        # x N) / |N|^2. Exact for a linear field on a planar polygon. Positions
        # and values from the first point, as the polyhedral gradient takes them.
        origin = cells.side_position(c, 0, 0)
        u0 = u[cells.side_point_id(c, 0, 0)]
        newell = tack.vec3(0.0, 0.0, 0.0)
        total = tack.vec3(0.0, 0.0, 0.0)
        for k in range(cells.num_faces(c)):
            a = cells.side_position(c, k, 0) - origin
            b = cells.side_position(c, k, 1) - origin
            newell += a.cross(b)
            total += (u[cells.side_point_id(c, k, 0)] + u[cells.side_point_id(c, k, 1)]
                      - 2.0 * u0) * (b - a)
        out[cells.entity_id(c)] = total.cross(newell) / newell.dot(newell)


@tack.kernel
def _zero_gradients(cells, out):
    for c in cells:
        out[cells.entity_id(c)] = [0.0, 0.0, 0.0]


@tack.kernel
def _entry_cells(offsets, cells):
    # Which cell each of cell_points()' entries belongs to.
    for c in range(offsets.shape[0] - 1):
        for k in range(offsets[c], offsets[c + 1]):
            cells[k] = c


def _polyhedral_scalar_gradients(data, field, at):
    """A scalar field's gradient on a topology without reference elements: per 3D
    cell Green-Gauss over the fanned faces, per polygon in its plane over its
    edges (both exact for a linear field), zero for polylines and for cell data
    (constant in each cell, as on the shape path), and at the points the mean of
    the cells' around each point."""
    cells = _vectors(data.num_cells, data.dtype)
    on_points = isinstance(field.space, H1)
    values = materialize(field.values) if on_points else None
    for group in data.launch_groups("cells", []):
        if not group.count:
            continue
        view = data.domain_view("cells", group)
        if on_points and group.shape.DIMENSION == 3:
            _polyhedral_gradients(view, values, cells)
        elif on_points and group.shape.DIMENSION == 2:
            _polygon_gradients(view, values, cells)
        else:
            _zero_gradients(view, cells)
    if at == "cells":
        return cells
    out = tack.empty((data.num_points,), tack.vec(3, data.dtype))
    out.fill(0)
    point_offsets, point_ids = data.topology.cell_points()
    entries = point_ids.shape[0]
    if entries:
        owners = tack.empty((entries,), data.id_dtype)
        _entry_cells(point_offsets, owners)
        offsets, links = data.topology.point_links()
        _average_links(offsets, links, _take(cells, owners), out)
    return out


@tack.kernel
def _row(values, k, out):
    for i in range(out.shape[0]):
        v = values[i]
        out[i] = tack.vec3(v[3 * k], v[3 * k + 1], v[3 * k + 2])


def _gradient_row(gradient, k):
    """Row ``k`` of a vector field's given gradient: its component ``k``'s."""
    if gradient is None:
        return None
    values = materialize(gradient.values)
    out = tack.empty((size_of(values),), tack.vec(3, arrays.dtype_of(values)))
    if out.shape[0]:
        _row(values, k, out)
    return Field(gradient.space, out, gradient.unit)


@tack.kernel
def _component(values, k, out):
    for i in range(out.shape[0]):
        out[i] = values[i][k]


@tack.kernel
def _place_rows(rows, k, out):
    for i in range(out.shape[0]):
        g = rows[i]
        out[i][3 * k] = g[0]
        out[i][3 * k + 1] = g[1]
        out[i][3 * k + 2] = g[2]


def _scalar_gradients(data, field, at):
    if isinstance(at, L2):
        # At an L2 space's nodes: each cell's own, from the cell alone.
        out = _vectors(at.size, data.dtype)
        groups = data.launch_groups("cells", [field], keys=(at,) if at.varies else ())
        for group, view in data._cell_node_views(at, groups):
            _node_gradients(view, field.view(group), out)
        return out
    if not getattr(data.topology, "reference_cells", True):
        return _polyhedral_scalar_gradients(data, field, at)
    if at == "cells":
        out = _vectors(data.num_cells, data.dtype)
        for_each(_gradients, data, "cells", field, out)
        return out
    # At the points: each cell's gradient at each of its corners, averaged over
    # the cells around each point, in layout order (VTK's and Viskores' point
    # gradients). Lines and vertices contribute zero.
    start_of, total = corner_layout(data.topology)
    corners = _vectors(total, data.dtype)
    singular = tack.empty((total,), tack.i32)
    for group in data.launch_groups("cells", [field]):
        if group.count:
            _corner_gradients(data.domain_view("cells", group), field.view(group),
                              start_of[group.root], corners, singular)
    out = tack.empty((data.num_points,), tack.vec(3, data.dtype))
    out.fill(0)
    if total:
        offsets, entries = data.topology.point_links()
        _average_links(offsets, entries, corners, out)
        if singular.max() > 0:              # rare: only then is any point rescaled
            _without_singular(offsets, entries, singular, corners, out, float("nan"))
    return out


@on_inputs_device
def gradients(data, field, at="cells"):
    """The gradient of ``field`` -- ``H1``, ``L2`` or ``Constant``, scalar or vector --
    at each cell's parametric center (``at="cells"``) or at the points
    (``at="points"``: each cell's gradient at the point, averaged over the cells
    around it, as VTK's vtkGradientFilter and Viskores' point gradients do).

    A scalar gives a 3-vector per entity; a vector of ``k`` components ``3k``
    values, row by row -- ``d(u_i)/d(x_j)`` at ``3 * i + j``, the layout of
    VTK's and Viskores' gradient tensors.

    ``at`` may instead be an ``L2`` space of the dataset: the gradient at each of
    its nodes in each cell -- a discontinuous field, exact at the nodes, so an
    ``H1`` field of order ``p`` keeps its order's detail in ``L2`` of order ``p``,
    where a gradient at the points averages it away. The field's basis gives its
    derivative in parametric coordinates and the geometry's Jacobian turns that
    into a world gradient, so the two need not share a space. A 2D cell's
    gradient lies in its plane (VTK's and Viskores' in-plane gradient), lines
    and vertices get zero, and a cell of no volume or area NaN (its Jacobian is
    singular). At the points, a pyramid's apex gives its gradient extrapolated from
    just below it, as vtkPyramid does; a cell whose Jacobian is singular at any
    other corner -- where two of its faces meet flat, as in a hexahedron with a
    straight corner -- has no gradient there (it grows without bound toward the
    corner) and is left out of that point's average; at a point no other cell
    gives a gradient, a mesh's corner say, the point takes the cell's gradient
    over its faces (Green-Gauss, bounded there) instead, and only a 2D cell or a
    solid as much inside out as not leaves the point NaN (VTK leaves its inverse
    Jacobian unset there; Viskores adds nothing but still counts the cell). A
    solid folded over (a corner's scaled Jacobian below -1e-3) takes its
    Green-Gauss gradient at every corner. ``flow_quantities`` derives divergence, vorticity and
    the Q-criterion from a vector field's gradient.

    A field given its gradient (``Field(..., gradient=)``) gives that instead of its
    basis', evaluated at the same points and averaged the same way.

    On a polyhedral topology there is no reference element. A cell's gradient
    of point data is then Green-Gauss over its boundary: the faces walked
    outward and fanned from their points' means (the triangles ``cell_geometry``
    measures the volume by), each triangle's area vector times the mean of its
    values, over the volume. That is exact for a linear field and needs no
    interpolant. Cell data, constant in each cell, has a zero gradient there,
    as on shapes. A polygon's is the same in its plane, over its edges; polylines
    get zero. At the points, the cells' gradients are averaged over the cells
    around each point."""
    field = _with_basis(data, _field(data, field))
    if not getattr(data.topology, "reference_cells", True) and field.gradient is not None:
        raise NotImplementedError("gradients of a field given its gradient on a mesh without "
                                  "reference elements")
    if (not getattr(data.topology, "reference_cells", True)
            and not isinstance(field.space, (H1, Constant))):
        raise NotImplementedError(f"gradients of a field on {field.space!r} on a mesh "
                                  "without reference elements: point or cell data only")
    if isinstance(at, L2):
        if at.topology is not data.topology:
            raise ValueError("gradients at the nodes of a space of another topology")
        if not getattr(data.topology, "reference_cells", True):
            raise NotImplementedError("gradients at an L2 space's nodes take cells with a "
                                      "reference element")
        space = at
    elif at not in ("cells", "points"):
        raise ValueError(f"gradients are at 'cells' or 'points', or an L2 space's nodes, "
                         f"not {at!r}")
    else:
        space = Values(data, at)
    # Per unit length; a difference of degrees Celsius is in kelvin.
    unit = units.derived(units.delta(field.unit), (data.unit, -1))
    width = width_of(field.values)
    if not width:
        return Field(space, _scalar_gradients(data, field, at), unit)
    n = space.size if isinstance(at, L2) else data.num_cells if at == "cells" \
        else data.num_points
    out = tack.empty((n,), tack.vec(3 * width, data.dtype))
    values = materialize(field.values)
    for k in range(width):
        part = tack.empty((size_of(field.values),), arrays.dtype_of(field.values))
        if part.shape[0]:
            _component(values, k, part)
        rows = _scalar_gradients(data, Field(field.space, part, gradient=_gradient_row(field.gradient, k)), at)
        if n:
            _place_rows(rows, k, out)
    return Field(space, out, unit)
