# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Cases the other modules did not reach: bound fields read through a conversion or kept by an
expression field, derivatives of clamp with field bounds and of atan2 at x = 0, empty fields,
and the arrays kept per topology."""

import gc
import weakref

import numpy as np
import pytest
import tack_expr as fx
from _expr_checks import assert_close, values
from tack_expr import parser as P

import tack
import tack.dataset as td


@tack.func
def f_lo(x):
    return x[0] * x[0]


@tack.func
def f_y(x):
    return x[1] + 1.0


@tack.func
def f_zero(x):
    return 0.0 * x[0]


@tack.func
def f_U(x):
    return tack.vec3(x[0] * x[1], x[1] * x[1], x[2] * x[0])


@pytest.fixture
def data(backend):
    d = td.uniform_grid((4, 4, 4), origin=(0.1, 0.2, 0.3), spacing=(0.3, 0.3, 0.3))
    d.fields["lo"] = td.interpolate(d, f_lo, td.H1(d, order=2))
    d.fields["y"] = td.interpolate(d, f_y, td.H1(d, order=2))
    d.fields["zero"] = td.interpolate(d, f_zero, td.H1(d, order=2))
    d.fields["U"] = td.interpolate(d, f_U, td.H1(d, order=2), dtype=tack.vec3())
    return d


def centers(data):
    return values("coords", data).reshape(-1, 3)


def test_compile_many_expands_derivatives_of_tensors(data):
    got = P.compile_many({"a": "div(grad(U))"}, data)(data)["a"].values.numpy()
    assert_close(got, values("div(grad(U))", data))


@pytest.mark.parametrize(("text", "which"), [
    ("clamp(lo, zero - 1, 10)", "lo"),          # lo active
    ("clamp(zero - 10, y, zero + 10)", "y"),    # x active
    ("clamp(-10, zero + 5, y)", "y"),           # hi active
])
def test_clamp_takes_the_derivative_of_the_active_bound(data, text, which):
    x = centers(data)
    ref = np.stack([2 * x[:, 0], 0 * x[:, 0], 0 * x[:, 0]], 1) if which == "lo" else np.tile([0.0, 1.0, 0.0], (len(x), 1))
    assert_close(values(f"grad({text})", data), ref)


def test_clamp_takes_the_second_derivative_of_the_active_bound(data):
    got = values("grad(grad(clamp(lo, zero - 1, 10))[0])", data).reshape(-1, 3)
    assert_close(got, np.tile([2.0, 0.0, 0.0], (len(got), 1)), tol=1e-3)


def test_atan2_second_derivative_is_finite_at_x_zero(data):
    got = values("grad(grad(atan2(y, zero))[1])", data)
    assert np.all(np.isfinite(got))
    assert_close(got, np.zeros_like(got), tol=1e-3)


def l2(data):
    return td.project(data, data.fields["lo"], td.L2(data, order=1))


def test_points_and_owner_read_a_bound_field(data):
    q = l2(data)
    with P.bound({"q": q}):
        points = fx.evaluate("points(q)", data, at="points").values.numpy()
        owner = fx.evaluate("owner(q)", data, at="faces").values.numpy()
    data.fields["q"] = q
    assert_close(points, values("points(q)", data, at="points"))
    assert_close(owner, values("owner(q)", data, at="faces"))


def test_an_expression_field_keeps_the_fields_it_was_bound_to(data):
    q = l2(data)
    data.fields["e"] = (q * 2).field(data)          # q is in no data.fields: bound as _f0
    data.fields["q"] = q
    assert_close(values("e + 1", data), values("q * 2 + 1", data))


def test_an_expression_field_is_carried_over_its_fields(data):
    data.fields["e"] = (data.fields["lo"] * 2).field(data)
    out = td.extract_cells(data, np.arange(10))
    assert_close(values("e", out), values("lo * 2", out))


def test_an_expression_field_not_carried_says_so(data):
    data.fields["e"] = (l2(data) * 2).field(data)
    with pytest.warns(UserWarning, match="not carried"):
        out = td.extract_cells(data, np.arange(10))
    assert "e" not in out.fields


@pytest.mark.parametrize(("dtype", "width"), [(tack.f32, 1), (tack.vec3(), 3), (tack.mat3(), 9)])
def test_an_empty_array_has_its_dtype_width(backend, dtype, width):
    assert P._width(tack.empty((0,), dtype)) == width
    assert P._width(tack.empty((2,), dtype)) == width


def test_update_ghosts_with_a_mesh_quantity(backend):
    pieces = td.ghost_cells(td.split(td.wavelet(10), 4))
    fx.evaluate("cells(RTData)", pieces, at="cells", name="C")
    got = fx.evaluate("measures * C", pieces, update_ghosts=True)
    assert len(got) == len(list(pieces))


def test_point_links_go_with_their_topology(backend):
    d = td.uniform_grid((4, 4, 4))
    d.fields["C"] = td.Field(td.Constant(d), tack.array(np.arange(d.num_cells, dtype=np.float32)))
    values("points(C)", d, at="points")
    assert "_tack_expr" in d.topology.__dict__
    kept = weakref.ref(d.topology)
    del d
    gc.collect()
    assert kept() is None
