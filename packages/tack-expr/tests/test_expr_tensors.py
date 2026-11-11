# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Rank-2 tensors: fields stored as mat3 or as 9 or 6 symmetric components, the velocity
gradient, and the tensor functions, against NumPy."""

import numpy as np
import pytest
import tack_expr as fx
from _expr_checks import assert_close, values

import tack
import tack.dataset as td


@tack.func
def f_v(x):
    return tack.vec3(x[1] * x[2] + 1.0, x[0] * x[0] - x[2], 2.0 * x[0] * x[1])


@pytest.fixture
def data(backend):
    d = td.uniform_grid((5, 4, 3), origin=(0.2, 0.3, 0.4), spacing=(0.25, 0.2, 0.3))
    d.fields["U"] = td.interpolate(d, f_v, td.H1(d, order=2), dtype=tack.vec3())
    rng = np.random.default_rng(3)
    n = d.num_cells
    a = rng.normal(size=(n, 3, 3)) + 3 * np.eye(3)
    s = 0.5 * (a + a.transpose(0, 2, 1))
    cells = td.Constant(d)
    m = tack.empty((n,), tack.mat3(d.dtype))
    tack.copyto(m, a)
    d.fields["A"] = td.Field(cells, m)
    six = tack.empty((n,), tack.vec(6, d.dtype))
    tack.copyto(six, np.stack([s[:, 0, 0], s[:, 1, 1], s[:, 2, 2], s[:, 0, 1], s[:, 1, 2], s[:, 0, 2]], axis=1))
    d.fields["S"] = td.Field(cells, six)
    nine = tack.empty((n,), tack.vec(9, d.dtype))
    tack.copyto(nine, a.reshape(n, 9))
    d.fields["A9"] = td.Field(cells, nine)
    d.a, d.s = a, s
    return d


def test_storage_forms(data):
    assert_close(values("A", data), data.a)
    assert_close(values("A9", data), data.a)
    assert_close(values("S", data), data.s)
    assert_close(values("S_XY + A_ZX", data), data.s[:, 0, 1] + data.a[:, 2, 0])
    assert_close(values("A[2][0] + S[1][2]", data), data.a[:, 2, 0] + data.s[:, 1, 2])
    assert_close(values("A[1]", data), data.a[:, 1])


def test_the_velocity_gradient(data):
    J = values("grad(U)", data)
    for i in range(3):
        assert_close(J[:, i], values(f"grad(U[{i}])", data))
    assert_close(values("trace(grad(U))", data), values("div(U)", data))
    assert_close(values("2 * skew(grad(U))[2][1]", data), values("curl(U)[0]", data))
    assert_close(values("grad(U * 2)", data), 2 * J)
    assert_close(values("grad(cross(U, coords))[0][1]", data), values("grad(cross(U, coords)[0])[1]", data))


def test_functions(data):
    a, s = data.a, data.s
    u = values("U", data)
    dev = s - np.trace(s, axis1=1, axis2=2)[:, None, None] / 3 * np.eye(3)
    checks = {
        "trace(A)": np.trace(a, axis1=1, axis2=2),
        "det(A)": np.linalg.det(a),
        "inv(A)": np.linalg.inv(a),
        "transpose(A)": a.transpose(0, 2, 1),
        "sym(A)": 0.5 * (a + a.transpose(0, 2, 1)),
        "dev(S)": dev,
        "vonmises(S)": np.sqrt(1.5 * np.einsum("nij,nij->n", dev, dev)),
        "eigvals(S)": np.linalg.eigvalsh(s)[:, ::-1],
        "ddot(A, S)": np.einsum("nij,nij->n", a, s),
        "mag(A)": np.linalg.norm(a, axis=(1, 2)),
        "outer(U, U)": np.einsum("ni,nj->nij", u, u),
        "dot(A, U)": np.einsum("nij,nj->ni", a, u),
        "dot(U, A)": np.einsum("ni,nij->nj", u, a),
        "dot(A, S)": a @ s,
        "2 * A - S / 4 + identity": 2 * a - s / 4 + np.eye(3),
        "{U, 2 * U, iHat}": np.stack([u, 2 * u, np.broadcast_to([1.0, 0.0, 0.0], u.shape)], axis=1),
        "if(trace(A) > 9, A, -A)": np.where((np.trace(a, axis1=1, axis2=2) > 9)[:, None, None], a, -a),
    }
    for text, want in checks.items():
        assert_close(values(text, data), want, tol=1e-3)


def test_tensors_with_the_operators_and_units(data):
    A = data.fields["A"]
    assert_close(fx.trace(fx.dot(A, A)).eval(data).values.numpy(), np.trace(data.a @ data.a, axis1=1, axis2=2),
                 tol=1e-3)
    assert_close((A * 2)[0, 1].eval(data).values.numpy(), 2 * data.a[:, 0, 1])
    got = fx.evaluate("vonmises(S)", data, fields={"S": "MPa"}, units="Pa")
    assert_close(got.values.numpy(), values("vonmises(S)", data) * 1e6, tol=1e-3)
    s = data.s
    for c, (i, j) in zip(("XX", "YY", "ZZ", "XY", "YZ", "XZ"), ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2))):
        data.fields[f"Stress{c}"] = td.Field(td.Constant(data), tack.array(np.ascontiguousarray(s[:, i, j])))
    got = fx.evaluate("eigvals(Stress)[0] - eigvals(Stress)[2]", data, fields=fx.cgns(), units="kPa")
    eig = np.linalg.eigvalsh(s)
    assert_close(got.values.numpy(), (eig[:, 2] - eig[:, 0]) * 1e-3, tol=1e-3)


@pytest.mark.parametrize(("text", "message"), [
    ("A * A", "use dot"),
    ("A + U", "only a tensor"),
    ("U / A", "only by a scalar"),
    ("trace(U)", "takes"),
    ("grad(A)", "not implemented"),
    ("curl(A)", "not implemented"),
])
def test_tensor_errors(data, text, message):
    with pytest.raises(fx.ExprError, match=message):
        fx.evaluate(text, data)


@tack.func
def f_stress(x):
    return tack.mat3(x[0] * x[0], x[0] * x[1], 0.0,
                     x[0] * x[1], x[1] * x[1] + x[2], x[2],
                     0.0, x[2], x[0] * x[2])


@pytest.fixture
def stressed(data):
    h2 = td.H1(data, order=2)
    data.fields["T"] = td.interpolate(data, f_stress, h2, dtype=tack.mat3())
    t = data.fields["T"].values.numpy()
    six = tack.empty((h2.size,), tack.vec(6, data.dtype))
    tack.copyto(six, np.stack([t[:, 0, 0], t[:, 1, 1], t[:, 2, 2], t[:, 0, 1], t[:, 1, 2], t[:, 0, 2]], axis=1))
    data.fields["T6"] = td.Field(h2, six)
    return data


def test_divergence_of_a_tensor_field(stressed):
    x = values("coords", stressed)
    want = np.stack([3 * x[:, 0], 3 * x[:, 1] + 1, x[:, 0]], axis=1)
    for text in ("div(T)", "div(T6)", "div(2 * T) / 2", "div(T + transpose(T)) / 2", "div(sym(T6))"):
        assert_close(values(text, stressed), want, tol=1e-3)
    assert_close(values("div(T) - div(T6)", stressed), 0 * want, tol=1e-3)


def test_derivatives_of_tensor_expressions(data):
    x = values("coords", data)
    U = np.stack([x[:, 1] * x[:, 2] + 1, x[:, 0] ** 2 - x[:, 2], 2 * x[:, 0] * x[:, 1]], axis=1)
    J = np.zeros((len(x), 3, 3))
    J[:, 0, 1], J[:, 0, 2] = x[:, 2], x[:, 1]
    J[:, 1, 0], J[:, 1, 2] = 2 * x[:, 0], -1.0
    J[:, 2, 0], J[:, 2, 1] = 2 * x[:, 1], 2 * x[:, 0]
    const = np.broadcast_to([0.0, 1.0, 0.0], x.shape)
    assert_close(values("div(sym(grad(U)))", data), const, tol=1e-3)
    assert_close(values("laplacian(U)", data), 2 * const, tol=1e-3)
    assert_close(values("div(outer(U, U))", data), np.einsum("nij,nj->ni", J, U), tol=1e-3)
    lame = "div(0.5 * div(U) * identity + 2 * sym(grad(U)))"
    assert_close(values(lame, data), 2 * const, tol=1e-3)
    assert_close((fx.div(fx.sym(fx.grad(data.fields["U"])))).eval(data).values.numpy(), const, tol=1e-3)
