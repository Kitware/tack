# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Fields as the one interface: a solver's own gradients through Field(..., gradient=),
expression fields stored under a name and read like stored fields, and their derivatives
in Tack's filters."""

import numpy as np
import pytest
import tack_expr as fx

import tack
import tack.dataset as td


@tack.func
def f_T(x):
    return x[0] + 2.0 * x[1] - 0.5 * x[2]


@tack.func
def f_V(x):
    return tack.vec3(x[0] * x[1], x[2] - x[0], x[1] * x[1])


def _grid():
    return td.uniform_grid((6, 5, 4), origin=(0.1, 0.2, 0.3), spacing=(0.3, 0.25, 0.4))


def _vectors(rows):
    rows = np.ascontiguousarray(rows, np.float32)
    out = tack.empty((len(rows),), tack.vec(rows.shape[1], tack.f32))
    tack.copyto(out, rows)
    return out


@pytest.fixture
def data(backend):
    d = _grid()
    d.fields["T"] = td.interpolate(d, f_T, td.H1(d))
    return d


def _values(text, data, **kw):
    return fx.evaluate(text, data, **kw).values.numpy()


# --- a solver's own gradient --------------------------------------------------------

def test_grad_of_a_field_given_its_gradient_is_the_given_one(data):
    given = np.random.default_rng(1).uniform(-1, 1, (data.num_cells, 3))
    T = data.fields["T"]
    data.fields["u"] = td.Field(T.space, T.values, gradient=td.Field(td.Constant(data), _vectors(given)))
    np.testing.assert_allclose(_values("grad(u)", data), given, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(_values("mag(grad(u)) * 2", data), 2 * np.linalg.norm(given, axis=1), rtol=1e-5)
    np.testing.assert_allclose(_values("grad(u * u)", data), 2 * _values("u", data)[:, None] * given,
                               rtol=1e-4, atol=1e-5)


def test_cell_data_given_its_gradient_has_one(data):
    given = np.tile([1.0, -2.0, 0.5], (data.num_cells, 1))
    data.fields["p"] = td.Field(td.Constant(data), tack.array(np.arange(data.num_cells, dtype=np.float32)),
                                gradient=td.Field(td.Constant(data), _vectors(given)))
    np.testing.assert_allclose(_values("grad(p)", data), given, rtol=1e-6)


def test_vector_derivatives_come_from_the_given_jacobian(data):
    J = np.random.default_rng(2).uniform(-1, 1, (data.num_cells, 3, 3))
    V = td.interpolate(data, f_V, td.H1(data), dtype=tack.vec3(tack.f32))
    data.fields["V"] = td.Field(V.space, V.values,
                                gradient=td.Field(td.Constant(data), _vectors(J.reshape(-1, 9))))
    np.testing.assert_allclose(_values("div(V)", data), np.trace(J, axis1=1, axis2=2), rtol=1e-5, atol=1e-6)
    curl = np.stack([J[:, 2, 1] - J[:, 1, 2], J[:, 0, 2] - J[:, 2, 0], J[:, 1, 0] - J[:, 0, 1]], 1)
    np.testing.assert_allclose(_values("curl(V)", data), curl, rtol=1e-5, atol=1e-6)
    S = 0.5 * (J + J.transpose(0, 2, 1))
    W = 0.5 * (J - J.transpose(0, 2, 1))
    q = 0.5 * ((W * W).sum((1, 2)) - (S * S).sum((1, 2)))
    np.testing.assert_allclose(_values("qcrit(V)", data), q, rtol=1e-4, atol=1e-6)


def test_second_derivatives_of_a_field_given_only_its_gradient_are_refused(data):
    T = data.fields["T"]
    data.fields["u"] = td.Field(T.space, T.values,
                                gradient=td.Field(td.Constant(data), _vectors(np.zeros((data.num_cells, 3)))))
    with pytest.raises(fx.ExprError, match="given its gradient"):
        fx.evaluate("laplacian(u)", data)


# --- expressions stored under a name ------------------------------------------------

def test_a_stored_expression_reads_as_a_field(data):
    data.fields["T2"] = fx.field("T * T + 1", data)
    T = _values("T", data, at="points")
    np.testing.assert_allclose(_values("T2 * 2", data, at="points"), 2 * (T * T + 1), rtol=1e-5)
    np.testing.assert_allclose(_values("grad(T2)", data), _values("grad(T * T + 1)", data), rtol=1e-5, atol=1e-6)
    data.fields["T3"] = fx.field("T2 - T", data)
    np.testing.assert_allclose(_values("T3", data, at="points"), T * T + 1 - T, rtol=1e-5, atol=1e-6)


def test_a_stored_expression_follows_its_inputs(data):
    data.fields["T2"] = fx.field("T * T + 1", data)
    T = data.fields["T"]
    data.fields["T"] = td.Field(T.space, tack.array(2.0 * T.values.numpy()))
    T = _values("T", data, at="points")
    np.testing.assert_allclose(_values("T2", data, at="points"), T * T + 1, rtol=1e-5)


def test_a_stored_expression_defined_by_itself_is_refused(data):
    data.fields["A"] = fx.field("T + 1", data)
    data.fields["T"] = fx.field("A * 2", data)
    with pytest.raises(fx.ExprError, match="in terms of itself"):
        fx.evaluate("T", data)


# --- expression fields in Tack's filters --------------------------------------------

@tack.kernel
def _hessians(cells, u, out):
    for c in cells:
        h = td.world_hessian(cells, u, c, cells.parametric_center())
        e = cells.entity_id(c)
        for a in range(3):
            for b in range(3):
                out[9 * e + 3 * a + b] = h[a, b]


def test_filters_differentiate_an_expression_field(data):
    data.fields["T2"] = fx.field("T * T + 1", data)
    np.testing.assert_allclose(td.gradients(data, "T2").values.numpy(), _values("grad(T * T + 1)", data),
                               rtol=1e-5, atol=1e-6)
    out = tack.zeros((9 * data.num_cells,), tack.f32)
    td.for_each(_hessians, data, "cells", data.fields["T2"], out)
    g = np.array([1.0, 2.0, -0.5])
    np.testing.assert_allclose(out.numpy().reshape(-1, 3, 3), np.broadcast_to(2 * np.outer(g, g), (data.num_cells, 3, 3)),
                               rtol=1e-4, atol=1e-4)


@tack.kernel
def _gradients_of(cells, u, out):
    for c in cells:
        out[cells.entity_id(c)] = td.world_gradient(cells, u, c, cells.parametric_center())


def test_a_kernel_of_your_own_differentiates_an_expression_field(data):
    f = fx.field("T * T", data)
    out = tack.zeros((data.num_cells,), tack.vec3(tack.f32))
    group = next(g for g in data.launch_groups("cells", [f]) if g.count)
    _gradients_of(data.domain_view("cells", group), f.view(group), out)
    np.testing.assert_allclose(out.numpy(), _values("grad(T * T)", data), rtol=1e-4, atol=1e-5)


# --- one resolver; carried and written ----------------------------------------------

def test_filters_take_an_expression(data):
    deferred = td.contour(data, fx.field("T * T"), 1.0)
    by_field = td.contour(data, fx.field("T * T", data), 1.0)
    assert deferred.num_points == by_field.num_points > 0
    np.testing.assert_allclose(td.gradients(data, fx.field("T * T")).values.numpy(),
                               _values("grad(T * T)", data), rtol=1e-4, atol=1e-5)
    with pytest.raises(KeyError):
        td.contour(data, "nosuch", 1.0)


def test_a_filter_output_has_the_expression_again(data):
    data.fields["e"] = fx.field("T * T", data)
    out = td.threshold(data, "T", 0.5, 2.0)
    assert 0 < out.num_cells < data.num_cells
    assert type(out.fields["e"].space).__name__ == "ExpressionSpace"
    np.testing.assert_allclose(_values("e", out), _values("T * T", out), rtol=1e-5)


def test_writers_store_an_expression_field(data):
    pytest.importorskip("vtkmodules")
    from tack.interop.vtk import dataset_to_vtk

    data.fields["e"] = fx.field("T * T", data)
    data.fields["g"] = fx.field("mag(grad(T))", data)
    grid = dataset_to_vtk(data)
    np.testing.assert_allclose(np.asarray(grid.point_data["e"]), _values("T * T", data, at="points"), rtol=1e-5)
    np.testing.assert_allclose(np.asarray(grid.cell_data["g"]), _values("mag(grad(T))", data), rtol=1e-5)


def test_reductions_of_an_expression_are_of_its_values(data):
    data.fields["stored"] = fx.evaluate("T * T + 1", data, at="points")
    by_text = td.statistics(data, fx.field("T * T + 1"))
    stored = td.statistics(data, "stored")
    assert by_text.n == stored.n
    np.testing.assert_allclose([by_text.min, by_text.max, by_text.mean], [stored.min, stored.max, stored.mean],
                               rtol=1e-6)
    np.testing.assert_allclose(td.entropy(data, fx.field("T * T + 1")), td.entropy(data, "stored"))
    # The integral is the expression's own, by quadrature through its view: exact for
    # T * T + 1, where its values at the points, interpolated, are not.
    g, w = np.polynomial.legendre.leggauss(3)
    axes = [((b - a) / 2 * g + (a + b) / 2, (b - a) / 2 * w) for a, b in ((0.1, 1.6), (0.2, 1.2), (0.3, 1.5))]
    x, y, z = np.meshgrid(*(p for p, _ in axes), indexing="ij")
    weights = np.einsum("i,j,k->ijk", *(q for _, q in axes))
    exact = float((weights * ((x + 2 * y - 0.5 * z) ** 2 + 1)).sum())
    got = td.integrate(data, {"e": fx.field("T * T + 1"), "f": fx.field("T * T + 1", data)}).values
    np.testing.assert_allclose([got["e"], got["f"]], exact, rtol=1e-5)


def test_an_expression_reduces_over_pieces(data):
    whole = td.statistics(data, fx.field("T * T + 1"))
    pieces = td.PartitionedDataSet([data, td.extract_cells(data, np.arange(data.num_cells // 2))])
    both = td.statistics(pieces, fx.field("T * T + 1"))
    assert both.n > whole.n
    assert both.max == pytest.approx(whole.max)
    flux = td.integrate(td.PartitionedDataSet([data]), {"t": fx.field("T * T + 1")}).values["t"]
    assert flux == pytest.approx(td.integrate(data, {"t": fx.field("T * T + 1")}).values["t"])


def test_a_surface_flux_and_a_component(data):
    data.fields["V"] = fx.evaluate("coords", data, at="points")
    box = td.extract_surface(data)
    # normals is derived from the surface where it is read, the cells here: nothing stores it.
    flux = td.integrate(box, {"flux": fx.field("dot(V, normals)")}).values["flux"]
    volume = td.integrate(data, {"div": fx.field("div(V)")}).values["div"]
    np.testing.assert_allclose(flux, volume, rtol=1e-5)                 # the divergence theorem
    np.testing.assert_allclose(td.statistics(data, fx.field("V_X")).mean, _values("coordsX", data, at="points").mean(),
                               rtol=1e-5)
    with pytest.raises(KeyError):
        td.statistics(data, "nosuch")


def test_normals_are_read_where_the_expression_is(data):
    box = td.extract_surface(data)
    made = td.surface_normals(box)
    np.testing.assert_allclose(_values("normals", box), made.fields["cell_normals"].values.numpy())
    np.testing.assert_allclose(_values("normals", box, at="points"), made.fields["normals"].values.numpy())
    np.testing.assert_allclose(_values("normals", data, at="faces"), td.face_geometry(data).normal.values.numpy())
    np.testing.assert_allclose(_values("measures", data).sum(), 0.3 * 0.25 * 0.4 * data.num_cells, rtol=1e-5)


def test_a_stored_field_comes_before_a_derived_one(data):
    box = td.extract_surface(data)
    flipped = -td.surface_normals(box).fields["cell_normals"].values.numpy()
    box.fields["normals"] = td.Field(td.Values(box, "cells"), _vectors(flipped))
    np.testing.assert_allclose(_values("normals", box), flipped)
    np.testing.assert_allclose(_values("mag(normals)", box, at="points"), 1.0, rtol=1e-6)
