# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Expressions on DG data as libmesh-style codes write it (#424): a quadratic equispaced
geometry, bent, with discontinuous fields (L2, orders 1-5, equispaced and Gauss-Lobatto)
on hexahedra, tetrahedra, wedges and a mesh mixing them. Results at the field's nodes and
at cells are held to Tack's own evaluation and gradients of the same field."""

import numpy as np
import pytest
import tack_expr as fx

import tack
import tack.dataset as td

_CORNERS = ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))
_WEDGES = ((0, 1, 3, 4, 5, 7), (1, 2, 3, 5, 6, 7))
_TETS = ((0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6), (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6))


@tack.func
def _bend(x):
    return [x[0] + 0.15 * x[1] * x[1], x[1] + 0.2 * x[0] * x[0], x[2] + 0.1 * x[0] * x[1]]


@tack.func
def _smooth(x):
    return tack.sin(2.0 * x[0]) * x[1] + tack.cos(x[2]) * x[0] * x[0]


def _tet_positive(tet, pts):
    a, b, c, d = (pts[i] for i in tet)
    return tet if np.dot(np.cross(b - a, c - a), d - a) > 0 else (tet[0], tet[2], tet[1], tet[3])


def _mesh(cells):
    """A 3 x 2 x 2 block of unit cubes, each a hexahedron, two wedges or six tetrahedra
    (by its index in a mesh mixing them), over shared points."""
    pts = np.array([[i, j, k] for k in range(3) for j in range(3) for i in range(4)], np.float64)
    pid = lambda i, j, k: i + 4 * (j + 3 * k)
    types, conn = [], []
    for n, (k, j, i) in enumerate(np.ndindex(2, 2, 3)):
        h = [pid(i + a, j + b, k + c) for a, b, c in _CORNERS]
        kind = {"hexahedra": 0, "wedges": 1, "tetrahedra": 2, "mixed": n % 3}[cells]
        if kind == 0:
            types.append(12)
            conn.append(h)
        elif kind == 1:
            for w in _WEDGES:
                types.append(13)
                conn.append([h[p] for p in w])
        else:
            for t in _TETS:
                types.append(10)
                conn.append([h[p] for p in _tet_positive(t, pts[h])])
    offsets = np.concatenate([[0], np.cumsum([len(c) for c in conn])])
    topology = td.UnstructuredTopology(np.array(types, np.uint8), offsets,
                                       np.concatenate(conn), num_points=len(pts))
    data = td.DataSet(topology, tack.array(pts * 0.5, tack.vec3(tack.f64)))
    data.geometry = td.interpolate(data, _bend, td.H1(data, order=2, basis="equispaced"),
                                   dtype=tack.vec3(tack.f64))
    return data


@pytest.fixture(scope="module", params=["hexahedra", "tetrahedra", "wedges", "mixed"])
def mesh(request):
    return request.param


@pytest.mark.parametrize("basis", ["equispaced", "gauss_lobatto"])
@pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
def test_dg_fields_on_a_quadratic_geometry(f64_backend, mesh, order, basis):
    data = _mesh(mesh)
    space = td.L2(data, order=order, basis=basis)
    try:
        _ = space.size
    except NotImplementedError as e:            # Tack's own refusal: no such nodes there
        pytest.skip(str(e))
    data.fields["u"] = td.interpolate(data, _smooth, space)
    u = data.fields["u"].values.numpy()

    at_nodes = fx.evaluate("2 * u - 1", data, at=space)
    assert at_nodes.space == space
    np.testing.assert_allclose(at_nodes.values.numpy(), 2 * u - 1, rtol=1e-13, atol=1e-13)

    expected = td.gradients(data, "u", at=space).values.numpy().reshape(-1, 3)
    got = fx.evaluate("grad(u)", data, at=space).values.numpy().reshape(-1, 3)
    np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-9)

    expected = td.gradients(data, "u").values.numpy().reshape(-1, 3)
    got = fx.evaluate("grad(u)", data, at="cells").values.numpy().reshape(-1, 3)
    np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-9)

    centers = td.values_at_centers(data, "u").values.numpy()
    np.testing.assert_allclose(fx.evaluate("u", data, at="cells").values.numpy(), centers,
                               rtol=1e-12, atol=1e-12)

    # At the points, a discontinuous field is the mean of the cells' corner values, as
    # project into H1 takes it, and only through points().
    at_points = td.project(data, "u", td.H1(data)).values.numpy()
    np.testing.assert_allclose(fx.evaluate("points(u)", data, at="points").values.numpy(),
                               at_points, rtol=1e-12, atol=1e-12)
    with pytest.raises(fx.ExprError, match="not continuous"):
        fx.evaluate("u", data, at="points")
