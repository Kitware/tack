# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Expressions read and written in each kind of space: higher-order nodes, faces, H1
targets, and the HDiv and HCurl vector spaces."""

import numpy as np
import pytest
import tack_expr as fx
from _expr_checks import assert_close, values, with_position_flux

import tack
import tack.dataset as td

pytestmark = pytest.mark.quick


@tack.func
def quadratic(x):
    return x[0] * x[0] + x[1] * x[2]


@pytest.fixture
def order2(backend):
    """u = x^2 + y z, held exactly by an order-2 H1 field."""
    data = td.uniform_grid((6, 5, 4), origin=(0.2, -0.5, 1.0), spacing=(0.3, 0.25, 0.4))
    data.fields["u"] = td.interpolate(data, quadratic, td.H1(data, order=2))
    return data


@pytest.fixture
def faces(backend):
    """The position's flux through each face, and a random cell field TC."""
    data = with_position_flux(td.uniform_grid((6, 5, 4), spacing=(0.5, 0.4, 0.3)))
    rng = np.random.default_rng(3)
    data.fields["TC"] = td.Field(td.Constant(data), tack.array(rng.random(data.num_cells, dtype=np.float32)))
    return data


def test_order2_values_at_l2_nodes(order2):
    l2 = td.L2(order2, order=2)
    x = values("coords", order2, at=l2)
    u = x[:, 0] ** 2 + x[:, 1] * x[:, 2]
    assert_close(values("u", order2, at=l2), td.project(order2, "u", l2).values.numpy())
    assert_close(values("u * u + 2", order2, at=l2), u * u + 2)
    assert_close(values("mag(grad(u))", order2, at=l2), np.sqrt((2 * x[:, 0]) ** 2 + x[:, 2] ** 2 + x[:, 1] ** 2))


def test_order2_gradient_at_cells(order2):
    x = values("coords", order2)
    got = values("grad(u)", order2)
    assert_close(got, td.gradients(order2, "u", at="cells").values.numpy())
    assert_close(got, np.stack([2 * x[:, 0], x[:, 2], x[:, 1]], axis=1))


def test_into_the_h1_space_of_a_field(order2):
    h1 = order2.fields["u"].space
    assert_close(values("u * 2 + 1", order2, at=h1), order2.fields["u"].values.numpy() * 2 + 1)


def test_face_sums(faces):
    assert_close(values("fsum(flux)", faces), td.net_flux(faces, "flux").values.numpy())
    volume = td.cell_geometry(faces).measure.values.numpy()
    assert_close(values("fsum(flux)", faces) / volume, np.full(faces.num_cells, 3.0))


@tack.kernel
def side_cells(faces, s0, s1):
    for f in faces:
        e = faces.entity_id(f)
        s0[e] = faces.side_cell(f, 0)
        s1[e] = faces.side_cell(f, 1)


def test_owner_and_neighbor_at_faces(faces):
    nf = faces.topology.faces().num_faces
    s0, s1 = tack.zeros((nf,), tack.i32), tack.zeros((nf,), tack.i32)
    td.for_each(side_cells, faces, "faces", s0, s1)
    s0, s1 = s0.numpy(), s1.numpy()
    tc = faces.fields["TC"].values.numpy()
    flux = faces.fields["flux"].values.numpy()
    jump = np.where(s1 >= 0, tc[np.maximum(s1, 0)] - tc[s0], 0.0)
    assert_close(values("neighbor(TC) - owner(TC)", faces, at="faces"), jump)
    assert_close(values("flux * owner(TC)", faces, at="faces"), flux * tc[s0])


@pytest.mark.parametrize(("expr", "at"), [("TC + 1", "points"), ("flux * 2", "cells"), ("TC * 2", "faces")])
def test_what_a_location_cannot_read_is_refused(faces, expr, at):
    with pytest.raises(fx.ExprError):
        fx.compile_expression(expr, faces, at)


def test_derivatives_at_points_are_refused(order2):
    with pytest.raises(fx.ExprError):
        fx.compile_expression("grad(u)", order2, "points")


def test_cell_data_into_h1_needs_project(faces):
    h1 = td.H1(faces, order=2)
    with pytest.raises(fx.ExprError):
        fx.evaluate("TC * 2", faces, at=h1)
    got = values("TC * 2", faces, at=h1, project=True)
    faces.fields["TC2"] = td.Field(td.Constant(faces), tack.array(faces.fields["TC"].values.numpy() * 2))
    assert_close(got, td.project(faces, "TC2", h1).values.numpy(), tol=1e-6)


@tack.func
def normal_linear(x):
    return tack.vec3(1.0 + 2.0 * x[0], 3.0 - x[1], 0.5 + 4.0 * x[2])


@tack.func
def raviart_thomas(x):
    return tack.vec3(1.0 + 2.0 * x[0], 3.0 + 2.0 * x[1], 0.5 + 2.0 * x[2])


@tack.kernel
def linear_fluxes(faces, centers, normals, areas, out):
    for f in faces:
        e = faces.entity_id(f)
        out[e] = normal_linear(centers[e]).dot(normals[e]) * areas[e]


@tack.kernel
def raviart_thomas_fluxes(faces, centers, normals, areas, out):
    for f in faces:
        e = faces.entity_id(f)
        out[e] = raviart_thomas(centers[e]).dot(normals[e]) * areas[e]


@tack.kernel
def constant_integrals(edges, out):
    for i in edges:
        x0 = edges.corner_position(i, 0)
        x1 = edges.corner_position(i, 1)
        out[edges.entity_id(i)] = tack.vec3(1.0, -2.0, 0.5).dot(x1 - x0)


@tack.kernel
def rotation_integrals(edges, out):
    for i in edges:
        x0 = edges.corner_position(i, 0)
        x1 = edges.corner_position(i, 1)
        m = 0.5 * (x0 + x1)
        out[edges.entity_id(i)] = tack.vec3(-m[1], m[0], 0.0).dot(x1 - x0)


def hdiv_field(data, kernel):
    g = td.face_geometry(data)
    flux = tack.empty((td.HDiv(data).size,), data.dtype)
    td.for_each(kernel, data, "faces", td.face_centers(data).values, g.normal.values, g.area.values, flux)
    return td.Field(td.HDiv(data), flux)


def hcurl_field(data, kernel):
    integrals = tack.empty((td.HCurl(data).size,), data.dtype)
    td.for_each(kernel, data, "edges", integrals)
    return td.Field(td.HCurl(data), integrals)


@pytest.fixture
def vector_spaces(backend):
    data = td.uniform_grid((6, 5, 4), origin=(0.3, -0.2, 1.0), spacing=(0.5, 0.3, 0.7))
    data.fields["F"] = hdiv_field(data, linear_fluxes)
    data.fields["G"] = hcurl_field(data, constant_integrals)
    data.fields["R"] = hcurl_field(data, rotation_integrals)
    return data


def test_hdiv_values_and_divergence(vector_spaces):
    x = values("coords", vector_spaces)
    assert_close(values("F", vector_spaces), np.stack([1 + 2 * x[:, 0], 3 - x[:, 1], 0.5 + 4 * x[:, 2]], axis=1))
    assert_close(values("div(F)", vector_spaces), np.full(vector_spaces.num_cells, 5.0))
    l1 = td.L2(vector_spaces, order=1)
    x1 = values("coords", vector_spaces, at=l1)
    assert_close(values("mag(F)", vector_spaces, at=l1),
                 np.linalg.norm(np.stack([1 + 2 * x1[:, 0], 3 - x1[:, 1], 0.5 + 4 * x1[:, 2]], axis=1), axis=1))


def test_hcurl_values_and_curl(vector_spaces):
    n = vector_spaces.num_cells
    assert_close(values("G * 2", vector_spaces), np.tile([2.0, -4.0, 1.0], (n, 1)))
    assert_close(values("curl(R)", vector_spaces), np.tile([0.0, 0.0, 2.0], (n, 1)))


def test_hdiv_on_tetrahedra(backend):
    pytest.importorskip("vtkmodules.vtkFiltersGeneral")
    from _expr_vtk import tetrahedra

    from tack.interop.vtk import vtk_to_dataset

    tets = vtk_to_dataset(tetrahedra())
    tets.fields["F"] = hdiv_field(tets, raviart_thomas_fluxes)
    x = values("coords", tets)
    assert_close(values("F", tets), np.stack([1 + 2 * x[:, 0], 3 + 2 * x[:, 1], 0.5 + 2 * x[:, 2]], axis=1))
    assert_close(values("div(F)", tets), np.full(tets.num_cells, 6.0))


def test_vector_spaces_at_points_are_refused(vector_spaces):
    with pytest.raises(fx.ExprError):
        fx.compile_expression("F", vector_spaces, at="points")
