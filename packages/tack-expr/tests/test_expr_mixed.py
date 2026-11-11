# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Expressions on a mixed mesh, polyhedra beside hexahedra in one topology: each cell
group read as cells of its kind are, against the same mesh of hexahedra alone."""

import numpy as np
import pytest
import tack_expr as fx

import tack
import tack.dataset as td

#: A hexahedron's faces in VTK's order of its points, each wound out of it.
_HEX_FACES = ((0, 4, 7, 3), (1, 2, 6, 5), (0, 1, 5, 4), (3, 7, 6, 2), (0, 3, 2, 1), (4, 5, 6, 7))
_POLYHEDRA = (0, 3, 5)          # no two share a face, so each face is stored once


def _hexes():
    pts = np.array([[i, j, k] for k in range(3) for j in range(3) for i in range(3)], np.float64)
    pid = lambda i, j, k: i + 3 * (j + 3 * k)
    conn = [[pid(i + a, j + b, k + c) for a, b, c in ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                                                      (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))]
            for k in range(2) for j in range(2) for i in range(2)]
    return pts, conn


def _with_fields(topology, pts):
    data = td.DataSet(topology, tack.array(pts, tack.vec3(tack.f64)))
    data.fields["s"] = td.Field(td.H1(data), tack.array((pts ** 2).sum(1)))
    data.fields["v"] = td.Field(td.H1(data), tack.array(pts, tack.vec3(tack.f64)))
    data.fields["C"] = td.Field(td.Constant(data), tack.array(np.arange(8, dtype=np.float64)))
    return data


def _meshes():
    pts, conn = _hexes()
    hexes = td.UnstructuredTopology(np.full(8, 12, np.uint8), np.arange(0, 65, 8),
                                    np.array(conn).ravel(), num_points=len(pts))
    types = np.full(8, 12, np.uint8)
    types[list(_POLYHEDRA)] = 42
    faces = [[conn[c][p] for p in face] for c in _POLYHEDRA for face in _HEX_FACES]
    face_offsets = np.concatenate([[0], np.cumsum([len(f) for f in faces])])
    per_cell = np.zeros(8, np.int64)
    per_cell[list(_POLYHEDRA)] = 6
    try:
        mixed = td.UnstructuredTopology(
            types, np.arange(0, 65, 8), np.array(conn).ravel(), num_points=len(pts),
            polyhedra=(face_offsets, np.concatenate(faces), np.concatenate([[0], np.cumsum(per_cell)]),
                       np.arange(len(faces)), np.zeros(len(faces), np.uint8)))
    except TypeError:
        pytest.skip("this Tack holds no polyhedra beside shaped cells")
    return _with_fields(hexes, pts), _with_fields(mixed, pts)


@pytest.mark.parametrize("text, at", [("cells(s)", "cells"), ("C * 2 + 1", "cells"),
                                      ("points(C)", "points"), ("mag(v) + s", "points")])
def test_expressions_on_shapes_beside_polyhedra(backend, text, at):
    hexes, mixed = _meshes()
    expected = fx.evaluate(text, hexes, at=at).values.numpy()
    np.testing.assert_allclose(fx.evaluate(text, mixed, at=at).values.numpy(), expected,
                               rtol=1e-12)


def test_an_expression_needing_a_basis_names_the_polyhedra(backend):
    _, mixed = _meshes()
    with pytest.raises(fx.ExprError, match="polyhedral cells have no parametric coordinates"):
        fx.evaluate("grad(s)", mixed, at="cells")
