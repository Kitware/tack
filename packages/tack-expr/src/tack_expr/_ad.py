# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Device functions for derivatives in generated kernels: rows and columns of 3x3
Jacobians, and the Jacobians of the vector operations. A Jacobian's row i is the
gradient of component i."""

import tack


@tack.func
def row(J, i):
    return tack.vec3(J[i, 0], J[i, 1], J[i, 2])


@tack.func
def col(J, k):
    return tack.vec3(J[0, k], J[1, k], J[2, k])


@tack.func
def rows3(g):
    """A scalar's gradient as a vector's Jacobian: every component has it."""
    return tack.mat3(g, g, g)


@tack.func
def diag_mul(a, J):
    """The Jacobian of a * v, component by component, for a constant vector a."""
    return tack.mat3(a[0] * row(J, 0), a[1] * row(J, 1), a[2] * row(J, 2))


@tack.func
def div_vv(a, Ja, b, Jb):
    """The Jacobian of a / b, component by component."""
    return tack.mat3((row(Ja, 0) * b[0] - a[0] * row(Jb, 0)) / (b[0] * b[0]),
                     (row(Ja, 1) * b[1] - a[1] * row(Jb, 1)) / (b[1] * b[1]),
                     (row(Ja, 2) * b[2] - a[2] * row(Jb, 2)) / (b[2] * b[2]))


@tack.func
def cross_jac(u, Ju, v, Jv):
    """The Jacobian of u x v: column k is du/dx_k x v + u x dv/dx_k."""
    c0 = col(Ju, 0).cross(v) + u.cross(col(Jv, 0))
    c1 = col(Ju, 1).cross(v) + u.cross(col(Jv, 1))
    c2 = col(Ju, 2).cross(v) + u.cross(col(Jv, 2))
    return tack.mat3(c0, c1, c2).transpose()


@tack.func
def unit_jac(v, J):
    """The Jacobian of v / |v|: (I - n n^T) J / |v|."""
    m = v.norm()
    n = v / m
    return (tack.identity(3) - n.outer_product(n)) @ J / m


try:
    from tack.dataset import world_gradient as _wg
except ImportError:                               # a Tack without the public name
    from tack.dataset.algorithms.gradients import _world_gradient as _wg


try:
    from tack.dataset import world_hessian, world_jet
except ImportError:
    world_hessian = world_jet = None

STENCIL = world_hessian is None


@tack.func
def stencil_hessian(cells, u, c, pc):
    """u's Hessian in the world: the world gradient differentiated along each parametric
    axis by a fourth-order central difference, then mapped by the inverse Jacobian. Exact
    for fields up to order 4 on cells with an affine map, fourth-order accurate on others."""
    h = 0.25
    D = 0.0 * tack.identity(3)
    for k in range(3):
        e = tack.vec3(1.0 if k == 0 else 0.0, 1.0 if k == 1 else 0.0, 1.0 if k == 2 else 0.0)
        dk = tack.vec3(0.0, 0.0, 0.0)
        for s in range(4):
            offset = -2.0 if s == 0 else (-1.0 if s == 1 else (1.0 if s == 2 else 2.0))
            weight = 1.0 if s == 0 else (-8.0 if s == 1 else (8.0 if s == 2 else -1.0))
            dk += weight * _wg(cells, u, c, pc + (offset * h) * e)
        D += (dk / (12.0 * h)).outer_product(e)
    H = D @ cells.geometry_jacobian(c, pc).inverse()
    return 0.5 * (H + H.transpose())



if STENCIL:
    world_hessian = stencil_hessian
