# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""First derivatives of expressions, against float64 central differences of the exact fields
and against td.gradients and td.flow_quantities.

u = x^2 + y z and v = 2x - y + 3z are scalars, U = (x y, y^2 + z, x - 2 y z) a vector, all
held exactly by order-2 H1 fields, so every expression's derivative is known.
"""

import numpy as np
import pytest
import tack_expr as fx
from _expr_checks import assert_close, assert_close_to_largest, values

import tack
import tack.dataset as td


@tack.func
def f_u(x):
    return x[0] * x[0] + x[1] * x[2]


@tack.func
def f_v(x):
    return 2.0 * x[0] - x[1] + 3.0 * x[2]


@tack.func
def f_U(x):
    return tack.vec3(x[0] * x[1], x[1] * x[1] + x[2], x[0] - 2.0 * x[2] * x[1])


def u(x):
    return x[..., 0] ** 2 + x[..., 1] * x[..., 2]


def v(x):
    return 2 * x[..., 0] - x[..., 1] + 3 * x[..., 2]


def U(x):
    return np.stack([x[..., 0] * x[..., 1], x[..., 1] ** 2 + x[..., 2], x[..., 0] - 2 * x[..., 2] * x[..., 1]], -1)


def jac(f, x, h=1e-5):
    """d f / d x_k by central differences: (..., 3) for a scalar f, (..., 3, 3) for a vector."""
    cols = []
    for k in range(3):
        e = np.zeros(3)
        e[k] = h
        cols.append((f(x + e) - f(x - e)) / (2 * h))
    return np.stack(cols, -1)


def div(F):
    return lambda x: np.trace(jac(F, x), axis1=-2, axis2=-1)


def curl(F):
    def c(x):
        J = jac(F, x)
        return np.stack([J[..., 2, 1] - J[..., 1, 2], J[..., 0, 2] - J[..., 2, 0], J[..., 1, 0] - J[..., 0, 1]], -1)
    return c


def qcrit(F):
    def q(x):
        J = jac(F, x)
        return (-0.5 * (J[..., 0, 0] ** 2 + J[..., 1, 1] ** 2 + J[..., 2, 2] ** 2)
                - (J[..., 0, 1] * J[..., 1, 0] + J[..., 0, 2] * J[..., 2, 0] + J[..., 1, 2] * J[..., 2, 1]))
    return q


def mag(a):
    return np.linalg.norm(a, axis=-1)


def apart(*pairs):
    """Where each pair of functions differs by more than 1e-3: away from a branch's switch."""
    return lambda x: np.all([np.abs(a(x) - b(x)) > 1e-3 for a, b in pairs], axis=0)


def one(_):
    return 1.0


def three(_):
    return 3.0


CASES = [
    ("grad(u * u)", lambda x: jac(lambda y: u(y) ** 2, x), None),
    ("grad(sqrt(u + 2) * v)", lambda x: jac(lambda y: np.sqrt(u(y) + 2) * v(y), x), None),
    ("grad(exp(-u) + sin(v))", lambda x: jac(lambda y: np.exp(-u(y)) + np.sin(v(y)), x), None),
    ("grad(u ^ 3 / (1 + v * v))", lambda x: jac(lambda y: u(y) ** 3 / (1 + v(y) ** 2), x), None),
    ("grad(atan2(u, v) + log10(u + 1))",
     lambda x: jac(lambda y: np.arctan2(u(y), v(y)) + np.log10(u(y) + 1), x), None),
    ("grad(mag(U))", lambda x: jac(lambda y: mag(U(y)), x), None),
    ("grad(dot(U, coords) - mag2(U))", lambda x: jac(lambda y: (U(y) * y).sum(-1) - (U(y) ** 2).sum(-1), x), None),
    ("grad(U_X * U_Z + coordsY)", lambda x: jac(lambda y: U(y)[..., 0] * U(y)[..., 2] + y[..., 1], x), None),
    ("grad(u > v ? u : v * v)", lambda x: jac(lambda y: np.where(u(y) > v(y), u(y), v(y) ** 2), x), apart((u, v))),
    ("grad(min(u, v) + clamp(1, u, 3))", lambda x: jac(lambda y: np.minimum(u(y), v(y)) + np.clip(u(y), 1, 3), x),
     apart((u, v), (u, one), (u, three))),
    ("div(U * u)", lambda x: div(lambda y: U(y) * u(y)[..., None])(x), None),
    ("div(cross(U, coords))", lambda x: div(lambda y: np.cross(U(y), y))(x), None),
    ("curl(U * v + coords * u)", lambda x: curl(lambda y: U(y) * v(y)[..., None] + y * u(y)[..., None])(x), None),
    ("curl(norm(U))", lambda x: curl(lambda y: U(y) / mag(U(y))[..., None])(x), None),
    ("qcrit(U + {u, v, u * v})", lambda x: qcrit(lambda y: U(y) + np.stack([u(y), v(y), u(y) * v(y)], -1))(x), None),
    ("qcrit(U / (1 + u))", lambda x: qcrit(lambda y: U(y) / (1 + u(y))[..., None])(x), None),
]


@pytest.fixture
def data(backend):
    d = td.uniform_grid((7, 6, 5), origin=(0.2, 0.3, 0.4), spacing=(0.25, 0.2, 0.3))
    d.fields["u"] = td.interpolate(d, f_u, td.H1(d, order=2))
    d.fields["v"] = td.interpolate(d, f_v, td.H1(d, order=2))
    d.fields["U"] = td.interpolate(d, f_U, td.H1(d, order=2), dtype=tack.vec3())
    return d


def site(data, where):
    return "cells" if where == "cells" else td.L2(data, order=2)


@pytest.mark.parametrize("where", ["cells", "L2 order-2 nodes"])
@pytest.mark.parametrize(("expr", "exact", "valid"), CASES, ids=[c[0] for c in CASES])
def test_derivative_matches_central_differences(data, where, expr, exact, valid):
    at = site(data, where)
    x = values("coords", data, at=at).astype(np.float64)
    assert_close_to_largest(values(expr, data, at=at), exact(x), 2e-5, None if valid is None else valid(x))


def test_gradients_match_td_gradients(data):
    x = values("coords", data).astype(np.float64)
    assert_close_to_largest(values("grad(u * 1)", data), td.gradients(data, "u").values.numpy(), 2e-5)
    gu = td.gradients(data, "u", at="cells").values.numpy()
    assert_close(values("grad(2 * u + 1)", data), 2 * gu)
    cu = values("cells(u)", data).astype(np.float64)
    assert_close_to_largest(values("grad(cells(u) * u)", data), cu[:, None] * jac(u, x), 2e-5)


def test_vector_derivatives_match_td_flow_quantities(data):
    fq = td.flow_quantities(data, td.gradients(data, "U", at="cells"))
    assert_close(values("div(U)", data), fq.divergence.values.numpy())
    assert_close(values("curl(U)", data), fq.vorticity.values.numpy())
    assert_close(values("qcrit(U)", data), fq.q_criterion.values.numpy())
    x = values("coords", data)
    assert_close(values("div(U)", data), x[:, 1])


@pytest.mark.parametrize("expr", ["grad(laplacian(u))", "div(u * v)"])
def test_what_has_no_derivative_here_is_refused(data, expr):
    with pytest.raises(fx.ExprError):
        fx.compile_expression(expr, data, at="cells")


def test_derivatives_need_a_cell(data):
    fx.compile_expression("grad(u + 1)", data, at="cells")
    with pytest.raises(fx.ExprError):
        fx.compile_expression("grad(u * v)", data, at="points")
