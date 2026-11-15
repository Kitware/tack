# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""``finite_differences``: the gradient (or the second derivatives) of a point or cell field on a structured grid -- an image, a rectilinear or a curvilinear grid -- by finite differences in index space of order 2, 4 or 6, taken to world axes through the grid's own differences (VTK's vtkGradientFilter on structured data, at order 2)."""

from fractions import Fraction
from math import factorial

import numpy as np

import tack
from tack import arrays
from tack.arrays import materialize, size_of, width_of
from tack.dataset import units
from tack.dataset.algorithms._common import _field, _with_basis
from tack.dataset.dataset import Field
from tack.dataset.ghosts import Ghosts, _has
from tack.dataset.spaces import H1, Constant, Values
from tack.dataset.topology import StructuredTopology
from tack.lang.array import SoaArray
from tack.runtime.dispatch import on_inputs_device

__all__ = ['finite_differences']

_ORDERS = (2, 4, 6)


def _weights(offsets, derivative):
    """The weights of the values at ``offsets`` (integers, in index steps) that give
    the ``derivative``-th derivative at offset 0, exact for every polynomial of
    degree below ``len(offsets)``: the Vandermonde system solved in rationals, so
    (-1/2, 0, 1/2) is exactly that. Zero where there are too few points."""
    n = len(offsets)
    if derivative >= n:
        return [0.0] * n
    rows = [[Fraction(o) ** k for o in offsets] + [Fraction(factorial(k) if k == derivative
                                                            else 0)]
            for k in range(n)]
    for col in range(n):                    # Gauss-Jordan; the offsets are distinct
        pivot = next(r for r in range(col, n) if rows[r][col] != 0)
        rows[col], rows[pivot] = rows[pivot], rows[col]
        rows[col] = [v / rows[col][col] for v in rows[col]]
        for r in range(n):
            if r != col and rows[r][col] != 0:
                factor = rows[r][col]
                rows[r] = [a - factor * b for a, b in zip(rows[r], rows[col])]
    return [float(rows[k][n]) for k in range(n)]


def _windows(order, derivative):
    """The points a stencil spans, inside and at the edges: ``(centered, edge)``. Order
    2 takes VTK's narrowest stencils at the edges -- a first derivative from two
    points, a second from three -- so its gradient is VTK's; orders 4 and 6 take
    one-sided stencils of their own order there."""
    centered = order + 1
    if order == 2:
        return centered, derivative + 1
    return centered, order + derivative


def _table(order, derivative, dtype):
    """The stencils of one derivative, by the visible points a position has on each
    side along its axis, up to ``cap`` each (``left * (cap + 1) + right``): each
    stencil's first offset, its length and its weights (``width`` of them, padded
    with zeros). The end of an axis and a hidden point bound a stencil alike, so
    one table serves every axis: the centered stencil where it fits, else the edge
    stencil as far as there are points, none where there is only the one."""
    centered, edge = _windows(order, derivative)
    width = max(centered, edge)
    cap = width - 1
    e = order // 2
    starts, lengths, weights = [], [], []
    for left in range(cap + 1):
        for right in range(cap + 1):
            reach = left + right + 1
            m = min(centered if left >= e and right >= e else edge, reach)
            first = min(max(-e, -left), right - m + 1)
            starts.append(first)
            lengths.append(m if reach > 1 else 0)
            weights.extend(_weights(list(range(first, first + m)), derivative)
                           + [0.0] * (width - m))
    return (tack.array(np.array(starts, np.int32), tack.i32),
            tack.array(np.array(lengths, np.int32), tack.i32),
            tack.array(np.array(weights, dtype.numpy_dtype), dtype), cap, width)


@tack.func
def _reach(marks, bit, masked, p, i, n, stride, cap):
    """The points on each side of point `p` (position `i` of `n` along an axis of
    `stride`) a stencil may take, up to `cap`: to the axis' end, or to the first
    point marked `bit` where `masked`."""
    left = min(i, cap)
    right = min(n - 1 - i, cap)
    if masked != 0:
        seen = 0
        while seen < left and (marks[p - (seen + 1) * stride] & bit) == 0:
            seen += 1
        left = seen
        seen = 0
        while seen < right and (marks[p + (seen + 1) * stride] & bit) == 0:
            seen += 1
        right = seen
    return left, right


@tack.func
def _apply(values, vs, vo, positions, p, stride, left, right, cap, starts, lengths,
           weights, span):
    """The derivative along one index axis at point `p`, with `left` and `right`
    points to take on either side, of a component of the field (point `q`'s at
    `values[q * vs + vo]`) and of the positions: `(du, dx)`."""
    c = min(left, cap) * (cap + 1) + min(right, cap)
    s = starts[c]
    du = weights[0] * 0.0
    dx = tack.vec3(du)
    for m in range(lengths[c]):
        w = weights[c * span + m]
        q = p + (s + m) * stride
        du += w * values[q * vs + vo]
        dx += w * positions[q]
    return du, dx


@tack.func
def _mixed(values, vs, vo, positions, marks, bit, masked, p, sa, la, ra, jb, nb, sb,
           cap, starts, lengths, weights, span):
    """The mixed second derivative along index axes `a` (stride `sa`, with `la` and
    `ra` points on either side of `p`) and `b`, of a component of the field (as
    `_apply` reads it) and of the positions: the derivative along `a` of the
    derivatives along `b`, each of those with the points it has itself, so that no
    stencil reaches a hidden point. Without marks it is the product of the two
    stencils."""
    ca = min(la, cap) * (cap + 1) + min(ra, cap)
    duv = weights[0] * 0.0
    dxv = tack.vec3(duv)
    for m in range(lengths[ca]):
        q = p + (starts[ca] + m) * sa
        lb, rb = _reach(marks, bit, masked, q, jb, nb, sb, cap)
        du, dx = _apply(values, vs, vo, positions, q, sb, lb, rb, cap, starts, lengths,
                        weights, span)
        w = weights[ca * span + m]
        duv += w * du
        dxv += w * dx
    return duv, dxv


@tack.func
def _unit(v):
    """`v` normalized, or `v` itself (zero) where it has no length: a collapsed axis
    leaves the frame singular, so `_metrics` gives zero, not NaN."""
    n = v.norm()
    s = n * 0.0
    if n != 0.0:
        s = 1.0 / n
    return v * s


@tack.func
def _completed(ja, jb, jc, fa, fb, fc):
    """The index axes' world derivatives, an axis with no derivative at the point
    (`f` 1: an axis of one point, or both neighbours hidden) replaced by unit
    normals to the others, so that the inverse gives gradients in the grid's plane
    or along its line: VTK takes the world axis of the same number instead, which
    is the same where the grid lies in a plane of world axes."""
    flat = fa + fb + fc
    if flat == 1:
        if fa != 0:
            ja = _unit(jb.cross(jc))
        elif fb != 0:
            jb = _unit(jc.cross(ja))
        else:
            jc = _unit(ja.cross(jb))
    elif flat == 2:
        line = ja
        if fb == 0:
            line = jb
        if fc == 0:
            line = jc
        t = _unit(line)
        other = tack.vec3(1.0, 0.0, 0.0)
        if abs(t[0]) > 0.5:
            other = tack.vec3(0.0, 1.0, 0.0)
        u = _unit(t.cross(other))
        v = t.cross(u)
        if fa == 0:
            jb = u
            jc = v
        elif fb == 0:
            jc = u
            ja = v
        else:
            ja = u
            jb = v
    return ja, jb, jc


@tack.func
def _metrics(ja, jb, jc):
    """The rows of the inverse of the matrix of columns `ja`, `jb`, `jc` (each index's
    gradient in the world), by VTK's cofactors; zero where it is singular, as VTK
    leaves the derivative there."""
    det = (ja[0] * jb[1] * jc[2] + ja[1] * jb[2] * jc[0] + ja[2] * jb[0] * jc[1]
           - ja[2] * jb[1] * jc[0] - ja[1] * jb[0] * jc[2] - ja[0] * jb[2] * jc[1])
    inv = det * 0.0
    if det != 0.0:
        inv = 1.0 / det
    ra = tack.vec3(inv * (jb[1] * jc[2] - jb[2] * jc[1]),
                   -inv * (jb[0] * jc[2] - jb[2] * jc[0]),
                   inv * (jb[0] * jc[1] - jb[1] * jc[0]))
    rb = tack.vec3(-inv * (ja[1] * jc[2] - ja[2] * jc[1]),
                   inv * (ja[0] * jc[2] - ja[2] * jc[0]),
                   -inv * (ja[0] * jc[1] - ja[1] * jc[0]))
    rc = tack.vec3(inv * (ja[1] * jb[2] - ja[2] * jb[1]),
                   -inv * (ja[0] * jb[2] - ja[2] * jb[0]),
                   inv * (ja[0] * jb[1] - ja[1] * jb[0]))
    return ra, rb, rc


@tack.kernel
def _gradient(values, vs, vo, width, comp, positions, marks, bit, masked, nan, nx, ny, nz,
              cap, starts, lengths, weights, span, count, out):
    for p in range(count):
        base = (p * width + comp) * 3
        if masked != 0 and (marks[p] & bit) != 0:
            out[base] = nan
            out[base + 1] = nan
            out[base + 2] = nan
        else:
            i = p % nx
            j = (p // nx) % ny
            k = p // (nx * ny)
            la, ra = _reach(marks, bit, masked, p, i, nx, 1, cap)
            lb, rb = _reach(marks, bit, masked, p, j, ny, nx, cap)
            lc, rc = _reach(marks, bit, masked, p, k, nz, nx * ny, cap)
            ua, ja = _apply(values, vs, vo, positions, p, 1, la, ra, cap, starts,
                            lengths, weights, span)
            ub, jb = _apply(values, vs, vo, positions, p, nx, lb, rb, cap, starts,
                            lengths, weights, span)
            uc, jc = _apply(values, vs, vo, positions, p, nx * ny, lc, rc, cap, starts,
                            lengths, weights, span)
            ja, jb, jc = _completed(ja, jb, jc, la + ra == 0, lb + rb == 0, lc + rc == 0)
            da, db, dc = _metrics(ja, jb, jc)
            g = da * ua + db * ub + dc * uc
            out[base] = g[0]
            out[base + 1] = g[1]
            out[base + 2] = g[2]


@tack.kernel
def _hessian(values, vs, vo, width, comp, positions, marks, bit, masked, nan, nx, ny, nz,
             cap, starts, lengths, weights, span, cap2, starts2, lengths2, weights2, span2,
             count, out):
    for p in range(count):
        base = (p * width + comp) * 9
        if masked != 0 and (marks[p] & bit) != 0:
            for c in range(9):
                out[base + c] = nan
        else:
            i = p % nx
            j = (p // nx) % ny
            k = p // (nx * ny)
            sy = nx
            sz = nx * ny
            # The reach of the wider stencils; the first derivatives' is within it.
            la, ra = _reach(marks, bit, masked, p, i, nx, 1, cap2)
            lb, rb = _reach(marks, bit, masked, p, j, ny, sy, cap2)
            lc, rc = _reach(marks, bit, masked, p, k, nz, sz, cap2)
            ua, ja = _apply(values, vs, vo, positions, p, 1, la, ra, cap, starts,
                            lengths, weights, span)
            ub, jb = _apply(values, vs, vo, positions, p, sy, lb, rb, cap, starts,
                            lengths, weights, span)
            uc, jc = _apply(values, vs, vo, positions, p, sz, lc, rc, cap, starts,
                            lengths, weights, span)
            uaa, xaa = _apply(values, vs, vo, positions, p, 1, la, ra, cap2, starts2,
                              lengths2, weights2, span2)
            ubb, xbb = _apply(values, vs, vo, positions, p, sy, lb, rb, cap2, starts2,
                              lengths2, weights2, span2)
            ucc, xcc = _apply(values, vs, vo, positions, p, sz, lc, rc, cap2, starts2,
                              lengths2, weights2, span2)
            uab, xab = _mixed(values, vs, vo, positions, marks, bit, masked, p, 1, la, ra,
                              j, ny, sy, cap, starts, lengths, weights, span)
            uac, xac = _mixed(values, vs, vo, positions, marks, bit, masked, p, 1, la, ra,
                              k, nz, sz, cap, starts, lengths, weights, span)
            ubc, xbc = _mixed(values, vs, vo, positions, marks, bit, masked, p, sy, lb,
                              rb, k, nz, sz, cap, starts, lengths, weights, span)
            ja, jb, jc = _completed(ja, jb, jc, la + ra == 0, lb + rb == 0, lc + rc == 0)
            da, db, dc = _metrics(ja, jb, jc)
            g = da * ua + db * ub + dc * uc
            # The chain rule: the index-space second derivatives less the grid's own
            # curvature along the gradient, then to world axes on both sides.
            m = tack.mat3([uaa - g.dot(xaa), uab - g.dot(xab), uac - g.dot(xac)],
                          [uab - g.dot(xab), ubb - g.dot(xbb), ubc - g.dot(xbc)],
                          [uac - g.dot(xac), ubc - g.dot(xbc), ucc - g.dot(xcc)])
            r = tack.mat3(da, db, dc)
            h = r.transpose() @ m @ r
            out[base] = h[0, 0]
            out[base + 1] = h[0, 1]
            out[base + 2] = h[0, 2]
            out[base + 3] = h[1, 0]
            out[base + 4] = h[1, 1]
            out[base + 5] = h[1, 2]
            out[base + 6] = h[2, 0]
            out[base + 7] = h[2, 1]
            out[base + 8] = h[2, 2]


def _grid(data, field):
    """Where ``field``'s values lie on ``data``'s structured grid: ``(sizes,
    positions, space)``, the sizes padded to three with ones."""
    topology = data.topology
    dims = tuple(topology.point_dims) + (1,) * (3 - len(topology.point_dims))
    if isinstance(field.space, H1):
        if field.space.order != 1:
            raise NotImplementedError(f"finite_differences of a field on {field.space!r}: "
                                      "point data (order 1) or cell data only")
        return dims, data.geometry.values, Values(data, "points")
    from tack.dataset.algorithms.cell_centers import cell_centers

    centers = cell_centers(data).values
    return tuple(max(n - 1, 1) for n in dims), centers, Values(data, "cells")


def _hidden(data, field):
    """The marks that hide values from the differences, the bit that does, and
    whether any value is hidden: hidden points for point data, hidden cells for cell
    data, as VTK's vtkGradientFilter reads them."""
    ghosts = data.ghosts
    point_data = isinstance(field.space, H1)
    bit = Ghosts.HIDDENPOINT if point_data else Ghosts.HIDDENCELL
    marks = None if ghosts is None else ghosts.points if point_data else ghosts.cells
    if not _has(marks, bit):
        return tack.zeros((1,), tack.u8), bit, 0
    return marks, bit, 1


@on_inputs_device
def finite_differences(data, field, order=2, derivative=1):
    """The gradient (``derivative=1``) or the second derivatives (``derivative=2``) of
    ``field`` -- point or cell data, scalar or vector -- on a structured grid: an
    image, a rectilinear or a curvilinear grid. Each is taken by finite differences
    along the grid's index axes, of the field and of the positions alike, and
    turned to world axes by the inverse of the positions' derivatives, so one rule
    serves every kind of grid; cell data is differenced over the cells' centers.

    ``order`` is the differences' order of accuracy: 2 (three points along each
    axis), 4 (five) or 6 (seven); on a smooth field the error falls as the
    spacing to that power. At order 2 the edges take the narrowest stencils, a
    first derivative from two points, as VTK's vtkGradientFilter does on images,
    rectilinear and curvilinear grids, so that the gradient is VTK's (and first
    order at the edges); orders 4 and 6 take one-sided stencils of their own order
    there. An axis shorter than a stencil takes every point it has. Mixed second
    derivatives take the derivative along one axis of the derivatives along the
    other.

    Hidden points (``Ghosts.HIDDENPOINT``, for point data) and hidden cells
    (``HIDDENCELL``, for cell data) hold no values: they bound the stencils as the
    grid's edges do, so a point beside one takes a one-sided stencil and a point
    between two along an axis has no derivative along it, as in VTK; their own
    derivatives are NaN (VTK writes 0). Other marks hide nothing: ghost layers
    widen the stencils at a piece's boundary, and the cells of an AMR box that a
    finer level covers (``REFINEDCELL``) hold values and are differenced like the
    rest.

    The gradient of a scalar is a 3-vector per point (or cell); of a vector of
    ``k`` components ``3k`` values, ``d(u_i)/d(x_j)`` at ``3 * i + j``, the layout
    of ``gradients``, so ``flow_quantities`` takes it. The second derivatives are
    ``9`` values per component, ``d2(u_i)/(d(x_j) d(x_k))`` at ``9 * i + 3 * j +
    k``: the Hessian, symmetric. On a curvilinear grid they include the grid's
    curvature (the chain rule's second term), so they are those of the field in
    the world, not along the grid's lines.

    A grid of two dimensions (or one) gives derivatives in its plane (along its
    line). Where the positions' derivatives are singular the derivatives are zero,
    as VTK leaves them. This differs from ``gradients(at="points")``, which
    averages each cell's derivative of the field's interpolant at the point: that
    is right for finite element data, these for data sampled on a grid."""
    if order not in _ORDERS:
        raise ValueError(f"finite_differences' order is 2, 4 or 6, not {order!r}")
    if derivative not in (1, 2):
        raise ValueError(f"finite_differences' derivative is 1 or 2, not {derivative!r}")
    if not isinstance(data.topology, StructuredTopology):
        raise TypeError("finite_differences takes a structured grid (an image, a rectilinear "
                        f"or a curvilinear grid), not {type(data.topology).__name__}")
    field = _with_basis(data, _field(data, field))
    if not isinstance(field.space, (H1, Constant)):
        raise NotImplementedError(f"finite_differences of a field on {field.space!r}: point "
                                  "data or cell data only")
    sizes, positions, space = _grid(data, field)
    width = width_of(field.values) or 1
    n = size_of(field.values)
    per = 3 if derivative == 1 else 9
    dtype = data.dtype
    out = tack.empty((n,), tack.vec(per * width, dtype))
    unit = units.derived(units.delta(field.unit), (data.unit, -derivative))
    if not n:
        return Field(space, out, unit)
    values = materialize(field.values)
    if isinstance(values, SoaArray):
        # Each component its own array: read it at stride 1.
        reads = [(c, 1, 0) for c in values.components]
    else:
        flat = values._scalar_view() if width_of(field.values) else values
        reads = [(flat, width, comp) for comp in range(width)]
    if arrays.dtype_of(field.values) != dtype:
        reads = [(v.astype(dtype), vs, vo) for v, vs, vo in reads]
    marks, bit, masked = _hidden(data, field)
    starts, lengths, weights, cap, span = _table(order, 1, dtype)
    nx, ny, nz = sizes
    target = out._scalar_view()
    nan = float("nan")
    if derivative == 2:
        starts2, lengths2, weights2, cap2, span2 = _table(order, 2, dtype)
    for comp, (flat, vs, vo) in enumerate(reads):
        if derivative == 1:
            _gradient(flat, vs, vo, width, comp, positions, marks, bit, masked, nan, nx, ny,
                      nz, cap, starts, lengths, weights, span, n, target)
        else:
            _hessian(flat, vs, vo, width, comp, positions, marks, bit, masked, nan, nx, ny,
                     nz, cap, starts, lengths, weights, span, cap2, starts2, lengths2,
                     weights2, span2, n, target)
    return Field(space, out, unit)
