# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Second derivatives against float64 central differences of the exact fields.

A field's Hessian comes from its basis' second derivatives on a Tack with ``world_hessian``,
and otherwise from a stencil on the world gradient that is exact for these fields. Either way
float32 results differ only by rounding, which second derivatives amplify to about 1e-4 of the
largest value; in float64 they agree to 1e-7.

u = x^3 + x y z + z^2 (order 3) and U = (x y, y^2 + z, x - 2 y z) (order 2) are held exactly
by their H1 fields on a uniform grid, whose cells map affinely.
"""

import numpy as np
import pytest
import tack_expr as fx
from _expr_checks import assert_close_to_largest, values

import tack
import tack.dataset as td


@tack.func
def f_u(x):
    return x[0] * x[0] * x[0] + x[0] * x[1] * x[2] + x[2] * x[2]


@tack.func
def f_U(x):
    return tack.vec3(x[0] * x[1], x[1] * x[1] + x[2], x[0] - 2.0 * x[2] * x[1])


def u(x):
    return x[..., 0] ** 3 + x[..., 0] * x[..., 1] * x[..., 2] + x[..., 2] ** 2


def U(x):
    return np.stack([x[..., 0] * x[..., 1], x[..., 1] ** 2 + x[..., 2], x[..., 0] - 2 * x[..., 2] * x[..., 1]], -1)


H = 1e-3


def d(f, k):
    e = np.zeros(3)
    e[k] = H
    return lambda x: (f(x + e) - f(x - e)) / (2 * H)


def grad(f):
    return lambda x: np.stack([d(f, k)(x) for k in range(3)], -1)


def lap(f):
    return lambda x: sum(d(d(f, k), k)(x) for k in range(3))


def div(F):
    return lambda x: sum(d(lambda y, k=k: F(y)[..., k], k)(x) for k in range(3))


def curl(F):
    def c(x):
        g = [[d(lambda y, i=i: F(y)[..., i], j)(x) for j in range(3)] for i in range(3)]
        return np.stack([g[2][1] - g[1][2], g[0][2] - g[2][0], g[1][0] - g[0][1]], -1)
    return c


def grad_x(f):
    return grad(lambda y: grad(f)(y)[..., 0])


CASES = [
    ("laplacian(u)", lap(u)),
    ("div(grad(u))", lap(u)),
    ("grad(grad(u)[0])", grad_x(u)),
    ("curl(grad(u))", curl(grad(u))),
    ("laplacian(u * u)", lap(lambda y: u(y) * u(y))),
    ("laplacian(exp(-u) + sin(U_Y))", lap(lambda y: np.exp(-u(y)) + np.sin(U(y)[..., 1]))),
    ("laplacian(mag(U))", lap(lambda y: np.linalg.norm(U(y), axis=-1))),
    ("div(grad(dot(U, coords) / (1 + u)))", lap(lambda y: (U(y) * y).sum(-1) / (1 + u(y)))),
    ("grad(div(U * u))", grad(div(lambda y: U(y) * u(y)[..., None]))),
    ("curl(curl(U))", curl(curl(U))),
    ("grad(mag2(grad(u)))", grad(lambda y: (grad(u)(y) ** 2).sum(-1))),
]


def grid(dtype=tack.f32):
    data = td.uniform_grid((6, 5, 4), origin=(0.2, 0.3, 0.4), spacing=(0.25, 0.2, 0.3), dtype=dtype)
    data.fields["u"] = td.interpolate(data, f_u, td.H1(data, order=3), dtype=dtype)
    return data


@pytest.fixture
def data(backend):
    d = grid()
    d.fields["U"] = td.interpolate(d, f_U, td.H1(d, order=2), dtype=tack.vec3())
    return d


@pytest.mark.parametrize("where", ["cells", "L2 order-2 nodes"])
@pytest.mark.parametrize(("expr", "exact"), CASES, ids=[c[0] for c in CASES])
def test_second_derivative_matches_central_differences(data, where, expr, exact):
    at = "cells" if where == "cells" else td.L2(data, order=2)
    x = values("coords", data, at=at).astype(np.float64)
    assert_close_to_largest(values(expr, data, at=at), exact(x), 3e-4)


def test_float64(f64_backend):
    data = grid(tack.f64)
    x = values("coords", data)
    assert_close_to_largest(values("laplacian(u)", data), lap(u)(x), 1e-7)
    assert_close_to_largest(values("grad(grad(u)[0])", data), grad_x(u)(x), 1e-6)


@pytest.mark.parametrize("expr", ["grad(laplacian(u))", "laplacian(qcrit(U))"])
def test_third_derivatives_are_refused(data, expr):
    with pytest.raises(fx.ExprError):
        fx.compile_expression(expr, data)


def test_surfaces_are_refused(backend):
    data = td.uniform_grid((5, 4))
    data.fields["u"] = td.interpolate(data, f_u, td.H1(data, order=2))
    with pytest.raises(fx.ExprError, match="solid"):
        fx.compile_expression("laplacian(u)", data)
