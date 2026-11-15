# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Expressions written with Python operators on td.Fields give what the same expressions in the
calculator syntax give."""

import numpy as np
import pytest
import tack_expr as fx
from _expr_checks import assert_close, values, vector_array

import tack
import tack.dataset as td


@tack.func
def f_t(x):
    return x[0] * x[0] + x[1] * x[2] + 0.5


@tack.func
def f_u(x):
    return tack.vec3(x[0] * x[1], x[1] * x[1] + x[2], x[0] - 2.0 * x[2] * x[1])


@tack.func
def f_v(x):
    return tack.vec3(1.0 + x[2], 2.0 - x[0] * x[1], 0.5 + x[1])


@tack.func
def f_w(x):
    # curl (y + 3, y + 1, 2x + 3 - z): no component reaches zero in the box
    return tack.vec3(x[1] * x[2] + 2.0 * x[2], x[0] * x[0] + 3.0 * x[0], x[0] + x[1] * x[1] * 0.5 + 3.0 * x[1])


@pytest.fixture
def data(backend):
    d = td.uniform_grid((6, 5, 4), origin=(0.2, 0.3, 0.4), spacing=(0.25, 0.2, 0.3))
    h2 = td.H1(d, order=2)
    d.fields["T"] = td.interpolate(d, f_t, h2)
    for name, f in (("U", f_u), ("V", f_v), ("W", f_w)):
        d.fields[name] = td.interpolate(d, f, h2, dtype=tack.vec3())
    return d


def fields(d):
    return d.fields["T"], d.fields["U"], d.fields["V"], d.fields["W"]


def test_berks_example(data):
    _, U, V, W = fields(data)
    expr = U + V / fx.curl(W)
    assert str(expr) == "(_f0 + (_f1 / curl(_f2)))"
    assert expr.render(data)[0] == "(U + (V / curl(W)))"
    assert_close(expr.eval(data).values.numpy(), values("U + V / curl(W)", data))


@pytest.mark.parametrize(("build", "text", "at"), [
    (lambda T, U, V, W: 2 - T * 3, "2 - T * 3", "points"),
    (lambda T, U, V, W: 1 / (T + 1e-5), "1 / (T + 1e-5)", "points"),
    (lambda T, U, V, W: -T ** 2, "-(T ^ 2)", "points"),
    (lambda T, U, V, W: T % 0.3 + abs(T - 1), "T % 0.3 + abs(T - 1)", "points"),
    (lambda T, U, V, W: np.sqrt(T) + np.sin(T) * 2 - np.arctan2(T, 2.0), "sqrt(T) + sin(T) * 2 - atan2(T, 2)", "points"),
    (lambda T, U, V, W: fx.where((T > 0.6) & ~(T > 1.2), T, 2 * T), "if((T > 0.6) and not(T > 1.2), T, 2 * T)",
     "points"),
    (lambda T, U, V, W: U[1] + U[2] * (V + 1).x, "U[1] + U[2] * (V + 1)[0]", "points"),
    (lambda T, U, V, W: fx.vec(T, 2 * T, 1) + fx.cross(U, V), "{T, 2 * T, 1} + cross(U, V)", "points"),
    (lambda T, U, V, W: fx.mag(U) * fx.dot(V, W) + fx.coords.x, "mag(U) * dot(V, W) + coordsX", "points"),
    (lambda T, U, V, W: fx.grad(T * T) + fx.curl(U * T), "grad(T * T) + curl(U * T)", "cells"),
    (lambda T, U, V, W: fx.laplacian(np.exp(-T)) + fx.qcrit(U), "laplacian(exp(-T)) + qcrit(U)", "cells"),
    (lambda T, U, V, W: fx.cells(T) * fx.div(U), "cells(T) * div(U)", "cells"),
])
def test_operators_match_the_calculator_syntax(data, build, text, at):
    got = build(*fields(data)).eval(data, at=at).values.numpy()
    assert_close(got, values(text, data, at=at))


def test_a_field_not_stored_in_the_dataset(data):
    T = data.fields["T"]
    doubled = td.Field(T.space, tack.array(T.values.numpy() * 2))
    expr = doubled + T
    assert expr.render(data)[0] == "(_f0 + T)"
    assert_close(expr.eval(data, at=T.space).values.numpy(), 3 * T.values.numpy())
    assert set(data.fields) == {"T", "U", "V", "W"}


def test_a_compiled_expression_reads_its_fields_or_refs_at_each_call(data):
    T = data.fields["T"]
    bound = (T * 2).compile(data, at=T.space)
    by_name = (fx.ref("T") * 2).compile(data, at=T.space)
    t = T.values.numpy()
    tack.copyto(T.values, t + 1)
    assert_close(bound(data).values.numpy(), 2 * (t + 1))
    data.fields["T"] = td.Field(T.space, tack.array(t * 0 + 5))
    assert_close(bound(data).values.numpy(), 2 * (t + 1))
    assert_close(by_name(data).values.numpy(), np.full_like(t, 10.0))


def test_compile_many_takes_expressions_and_text(data):
    T, U, _, _ = fields(data)
    many = fx.compile_many({"a": fx.mag(fx.curl(U)), "b": "qcrit(U)", "c": fx.grad(T) * T}, data)
    out = many(data)
    assert_close(out["a"].values.numpy(), values("mag(curl(U))", data))
    assert_close(out["b"].values.numpy(), values("qcrit(U)", data))
    assert_close(out["c"].values.numpy(), values("grad(T) * T", data))


def test_lazy_fields_and_contours(data):
    T, U, _, _ = fields(data)
    expr = T * T + fx.mag2(U) * 0.1
    surface = td.contour(data, expr.field(data), 1.0)
    stored = td.contour(data, fx.field("T * T + mag2(U) * 0.1"), 1.0)
    assert (surface.num_cells, surface.num_points) == (stored.num_cells, stored.num_points)
    f = expr.field(data)
    assert f.space.continuous
    assert set(data.fields) == {"T", "U", "V", "W"}


def test_a_function_in_a_kernel(data):
    T, U, _, _ = fields(data)

    @tack.kernel
    def take(cells, q, out):
        for c in cells:
            out[cells.entity_id(c)] = q.value(cells, c, cells.parametric_center())

    out = tack.zeros((data.num_cells,))
    fx.for_each(take, data, (fx.mag(U) * T).function(data), out)
    assert_close(out.numpy(), values("mag(U) * T", data))


def test_fields_of_other_topologies_are_refused(data):
    other = td.uniform_grid((3, 3, 3))
    other.fields["T"] = td.Field(td.H1(other), tack.zeros((other.num_points,)))
    with pytest.raises(ValueError, match="another topology"):
        (data.fields["T"] + other.fields["T"]).eval(data)


def test_pieces_take_refs(backend):
    whole = td.wavelet(8)
    pieces = td.split(whole, 2, ghost_layers=1)
    with pytest.raises(ValueError, match="ref"):
        (whole.fields["RTData"] * 2).eval(pieces)
    got = (fx.ref("RTData") * 2 + 1).eval(pieces, at="points", name="r")
    assert len(got) == 2
    assert td.statistics(pieces, "r").max == pytest.approx(td.statistics(whole, "RTData").max * 2 + 1, rel=1e-6)


def test_no_truth_value(data):
    T = data.fields["T"]
    with pytest.raises(TypeError, match="where"):
        bool(T > 0.5)
    with pytest.raises(TypeError, match="where"):
        T < 0.5 < T  # noqa: B015


def test_what_cannot_be_in_an_expression(data):
    T = data.fields["T"]
    with pytest.raises(TypeError):
        T + np.ones(3)
    with pytest.raises(ValueError, match="finite"):
        T + float("inf")
    with pytest.raises(TypeError, match="field"):
        fx.cells(T * 2)


def test_vector_values_round_trip(backend):
    d = td.uniform_grid((4, 4, 4))
    rng = np.random.default_rng(3)
    d.fields["V"] = td.Field(td.H1(d), vector_array(rng.random((d.num_points, 3))))
    V = d.fields["V"]
    assert_close((V * 2 - V).eval(d, at="points").values.numpy(), V.values.numpy())
