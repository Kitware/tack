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


def _classes(n, order):
    """The positions along an axis of ``n`` points whose stencils differ: each edge
    position, and one for every interior position (all of them, when the axis is
    too short to have an interior)."""
    e = order // 2
    if n <= 2 * e + 1:
        return list(range(n))
    return [*range(e), e, *range(n - e, n)]


def _table(n, order, derivative, width):
    """Per stencil class along an axis of ``n`` points: its first offset, its length
    and its weights (``width`` of them, padded with zeros)."""
    centered, edge = _windows(order, derivative)
    e = order // 2
    starts, lengths, weights = [], [], []
    for i in _classes(n, order):
        m = min(centered if e <= i <= n - 1 - e else edge, n)
        first = min(max(i - e, 0), n - m)
        offsets = list(range(first - i, first - i + m))
        starts.append(first - i)
        lengths.append(m if n > 1 else 0)
        weights.extend(_weights(offsets, derivative) + [0.0] * (width - m))
    return starts, lengths, weights


def _tables(sizes, order, derivative, dtype):
    """The three axes' tables, one array each of starts, lengths and weights, with the
    number of classes per axis (``classes``) and of weights per class (``width``)."""
    centered, edge = _windows(order, derivative)
    width = max(centered, edge)
    classes = max(len(_classes(n, order)) for n in sizes)
    starts = np.zeros(3 * classes, np.int32)
    lengths = np.zeros(3 * classes, np.int32)
    weights = np.zeros(3 * classes * width, np.float64)
    for axis, n in enumerate(sizes):
        s, m, w = _table(n, order, derivative, width)
        starts[axis * classes:axis * classes + len(s)] = s
        lengths[axis * classes:axis * classes + len(m)] = m
        weights[axis * classes * width:axis * classes * width + len(w)] = w
    return (tack.array(starts, tack.i32), tack.array(lengths, tack.i32),
            tack.array(weights.astype(dtype.numpy_dtype), dtype), classes, width)


@tack.func
def _class(i, n, e):
    """The stencil class of position `i` along an axis of `n` points (`_classes`)."""
    c = e
    if n <= 2 * e + 1 or i < e:
        c = i
    elif i >= n - e:
        c = i - (n - 2 * e - 1)
    return c


@tack.func
def _axis(values, width, comp, positions, p, i, n, stride, axis, e, starts, lengths,
          weights, classes, span):
    """The derivative along one index axis at point `p` (position `i` of `n`), of the
    field's component `comp` and of the positions: `(du, dx)`."""
    c = axis * classes + _class(i, n, e)
    s = starts[c]
    du = weights[0] * 0.0
    dx = tack.vec3(du)
    for m in range(lengths[c]):
        w = weights[c * span + m]
        q = p + (s + m) * stride
        du += w * values[q * width + comp]
        dx += w * positions[q]
    return du, dx


@tack.func
def _cross_axes(values, width, comp, positions, p, ia, na, sa, a, ib, nb, sb, b, e, starts,
                lengths, weights, classes, span):
    """The mixed second derivative along index axes `a` and `b` at point `p`, of the
    field's component `comp` and of the positions: the product of the two axes'
    first-derivative stencils."""
    ca = a * classes + _class(ia, na, e)
    cb = b * classes + _class(ib, nb, e)
    duv = weights[0] * 0.0
    dxv = tack.vec3(duv)
    for m in range(lengths[ca]):
        wa = weights[ca * span + m]
        qa = p + (starts[ca] + m) * sa
        for l in range(lengths[cb]):
            w = wa * weights[cb * span + l]
            q = qa + (starts[cb] + l) * sb
            duv += w * values[q * width + comp]
            dxv += w * positions[q]
    return duv, dxv


@tack.func
def _completed(ja, jb, jc, na, nb, nc):
    """The index axes' world derivatives, an axis of one point (`n` 1, no derivative)
    replaced by unit normals to the others, so that the inverse gives gradients in
    the grid's plane or along its line: VTK takes the world axis of the same number
    instead, which is the same where the grid lies in a plane of world axes."""
    flat = 0
    if na == 1:
        flat += 1
    if nb == 1:
        flat += 1
    if nc == 1:
        flat += 1
    if flat == 1:
        if na == 1:
            ja = jb.cross(jc).normalized()
        elif nb == 1:
            jb = jc.cross(ja).normalized()
        else:
            jc = ja.cross(jb).normalized()
    elif flat == 2:
        line = ja
        if nb != 1:
            line = jb
        if nc != 1:
            line = jc
        t = line.normalized()
        other = tack.vec3(1.0, 0.0, 0.0)
        if abs(t[0]) > 0.5:
            other = tack.vec3(0.0, 1.0, 0.0)
        u = t.cross(other).normalized()
        v = t.cross(u)
        if na != 1:
            jb = u
            jc = v
        elif nb != 1:
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
def _gradient(values, width, comp, positions, nx, ny, nz, e, starts, lengths, weights,
              classes, span, count, out):
    for p in range(count):
        i = p % nx
        j = (p // nx) % ny
        k = p // (nx * ny)
        ua, ja = _axis(values, width, comp, positions, p, i, nx, 1, 0, e, starts, lengths,
                       weights, classes, span)
        ub, jb = _axis(values, width, comp, positions, p, j, ny, nx, 1, e, starts, lengths,
                       weights, classes, span)
        uc, jc = _axis(values, width, comp, positions, p, k, nz, nx * ny, 2, e, starts,
                       lengths, weights, classes, span)
        ja, jb, jc = _completed(ja, jb, jc, nx, ny, nz)
        ra, rb, rc = _metrics(ja, jb, jc)
        g = ra * ua + rb * ub + rc * uc
        base = (p * width + comp) * 3
        out[base] = g[0]
        out[base + 1] = g[1]
        out[base + 2] = g[2]


@tack.kernel
def _hessian(values, width, comp, positions, nx, ny, nz, e, starts, lengths, weights,
             starts2, lengths2, weights2, classes, span, span2, count, out):
    for p in range(count):
        i = p % nx
        j = (p // nx) % ny
        k = p // (nx * ny)
        sy = nx
        sz = nx * ny
        ua, ja = _axis(values, width, comp, positions, p, i, nx, 1, 0, e, starts, lengths,
                       weights, classes, span)
        ub, jb = _axis(values, width, comp, positions, p, j, ny, sy, 1, e, starts, lengths,
                       weights, classes, span)
        uc, jc = _axis(values, width, comp, positions, p, k, nz, sz, 2, e, starts, lengths,
                       weights, classes, span)
        uaa, xaa = _axis(values, width, comp, positions, p, i, nx, 1, 0, e, starts2,
                         lengths2, weights2, classes, span2)
        ubb, xbb = _axis(values, width, comp, positions, p, j, ny, sy, 1, e, starts2,
                         lengths2, weights2, classes, span2)
        ucc, xcc = _axis(values, width, comp, positions, p, k, nz, sz, 2, e, starts2,
                         lengths2, weights2, classes, span2)
        uab, xab = _cross_axes(values, width, comp, positions, p, i, nx, 1, 0, j, ny, sy, 1,
                               e, starts, lengths, weights, classes, span)
        uac, xac = _cross_axes(values, width, comp, positions, p, i, nx, 1, 0, k, nz, sz, 2,
                               e, starts, lengths, weights, classes, span)
        ubc, xbc = _cross_axes(values, width, comp, positions, p, j, ny, sy, 1, k, nz, sz, 2,
                               e, starts, lengths, weights, classes, span)
        ja, jb, jc = _completed(ja, jb, jc, nx, ny, nz)
        ra, rb, rc = _metrics(ja, jb, jc)
        g = ra * ua + rb * ub + rc * uc
        # The chain rule: the index-space second derivatives less the grid's own
        # curvature along the gradient, then to world axes on both sides.
        m = tack.mat3([uaa - g.dot(xaa), uab - g.dot(xab), uac - g.dot(xac)],
                      [uab - g.dot(xab), ubb - g.dot(xbb), ubc - g.dot(xbc)],
                      [uac - g.dot(xac), ubc - g.dot(xbc), ucc - g.dot(xcc)])
        r = tack.mat3(ra, rb, rc)
        h = r.transpose() @ m @ r
        base = (p * width + comp) * 9
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


def _refuse_hidden(data, field):
    ghosts = data.ghosts
    if ghosts is None:
        return
    if isinstance(field.space, H1):
        marks, bit, what = ghosts.points, Ghosts.HIDDENPOINT, "hidden points"
    else:
        marks, bit, what = ghosts.cells, Ghosts.HIDDENCELL, "hidden cells"
    if _has(marks, bit):
        raise NotImplementedError(f"finite_differences on a grid with {what}: their "
                                  "stencils would read values that are not there")


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
    derivatives take the product of the two axes' first-derivative stencils.

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
    is right for finite element data, these for data sampled on a grid. A grid
    with hidden points (for point data) or hidden cells (for cell data) is
    refused."""
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
    _refuse_hidden(data, field)
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
    flat = values._scalar_view() if width_of(field.values) else values
    if arrays.dtype_of(field.values) != dtype:
        flat = flat.astype(dtype)
    starts, lengths, weights, classes, span = _tables(sizes, order, 1, dtype)
    e = order // 2
    nx, ny, nz = sizes
    target = out._scalar_view()
    if derivative == 2:
        starts2, lengths2, weights2, _, span2 = _tables(sizes, order, 2, dtype)
    for comp in range(width):
        if derivative == 1:
            _gradient(flat, width, comp, positions, nx, ny, nz, e, starts, lengths, weights,
                      classes, span, n, target)
        else:
            _hessian(flat, width, comp, positions, nx, ny, nz, e, starts, lengths, weights,
                     starts2, lengths2, weights2, classes, span, span2, n, target)
    return Field(space, out, unit)
