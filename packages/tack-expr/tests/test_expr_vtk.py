# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""tack_expr against VTK: vtkArrayCalculator's syntax, VTK's conversions between points and
cells, tetrahedra, polyhedra, float64 and invalid values."""

import numpy as np
import pytest

pytest.importorskip("vtkmodules.vtkFiltersCore")

import tack_expr as fx
from _expr_checks import assert_close, values, with_position_flux
from _expr_vtk import array, calc, grids, tetrahedra
from vtkmodules.vtkFiltersCore import vtkCellDataToPointData, vtkPointDataToCellData

import tack
import tack.dataset as td
from tack.interop.vtk import vtk_to_dataset

pytestmark = pytest.mark.quick

SYNTAX = [
    "T + P", "T - 2*P", "T * P / (P + 1)", "T ^ 2", "2 ^ 3 ^ 2", "-T ^ 2", "T + P ^ 2",
    "T % 0.3", "-T % 0.3", "(T + 1) * -P", "abs(T - 0.5)", "acos(T)", "asin(T)", "atan(T)",
    "ceil(10*T)", "floor(10*T)", "cos(T)", "cosh(T)", "exp(T)", "ln(P + 1)", "log10(P + 1)",
    "sin(T)", "sinh(T)", "sqrt(T)", "tan(T)", "tanh(T)", "sign(T - 0.5)", "min(T, P)",
    "max(T, P)", "pow(T, 3)", "atan2(T, P)", "clamp(0.2, T, 0.8)", "pi * T", "V_X + V_Y * V_Z",
    "coordsX + coordsY - coordsZ", "T > 0.5 ? T : P", "if(T > 0.5, T, P)",
    "(T > 0.3) and (P < 0.7)", "(T < 0.5) | (P > 0.5)", "not(T > 0.5)", "T = T", "T <> P",
    "mag(V)", "dot(V, W)", "V + W", "2 * V", "V * T", "V / 2", "cross(V, W)", "norm(V)",
    "iHat * T + jHat * P + kHat", "coords", "{T, P, 1}", "V * W", "V[1]", "T > 0.5 ? V : W",
    "mag(cross(V, W)) / (mag(V) * mag(W))",
]


@pytest.mark.parametrize("expr", SYNTAX)
def test_point_data_matches_vtk_array_calculator(backend, expr):
    points, _ = grids()
    assert_close(values(expr, vtk_to_dataset(points), at="points"), calc(points, expr))


@pytest.mark.parametrize("expr", ["T + P * 2", "mag(V) * T", "T > 0.5 ? V : W", "cross(V, W) / (1 + T)"])
def test_cell_data_matches_vtk_array_calculator(backend, expr):
    _, cells = grids()
    assert_close(values(expr, vtk_to_dataset(cells)), calc(cells, expr, cells=True))


def test_point_data_at_cells_matches_vtk_point_data_to_cell_data(backend):
    points, _ = grids()
    data = vtk_to_dataset(points)
    ref = array(vtkPointDataToCellData()(points), "T", cells=True)
    assert_close(values("cells(T)", data), ref)
    assert_close(values("T", data), ref)


def test_cell_data_at_points_matches_vtk_cell_data_to_point_data(backend):
    _, cells = grids()
    data = vtk_to_dataset(cells)
    out = vtkCellDataToPointData()(cells)
    assert_close(values("points(T)", data, at="points"), array(out, "T"))
    assert_close(values("points(V) * 2", data, at="points"), 2 * array(out, "V"))
    compiled = fx.compile_expression("points(T) * 2 + points(P)", data, at="points")
    assert compiled.mode == "pointloop"
    assert_close(compiled(data).values.numpy(), 2 * array(out, "T") + array(out, "P"))


def test_tetrahedra(backend):
    tets_vtk = tetrahedra()
    tets = vtk_to_dataset(tets_vtk)
    assert_close(values("T * 2 + P", tets, at="points"), calc(tets_vtk, "T * 2 + P"))
    assert_close(values("cells(T)", tets), array(vtkPointDataToCellData()(tets_vtk), "T", cells=True))
    assert_close(values("grad(T)", tets), td.gradients(tets, "T", at="cells").values.numpy())


def test_node_outputs_on_tetrahedra(backend):
    tets = vtk_to_dataset(tetrahedra())
    l2 = td.L2(tets, order=2, basis="equispaced")
    assert_close(values("T", tets, at=l2), td.project(tets, "T", l2).values.numpy())
    l1 = td.L2(tets, order=1)
    t, p = (td.project(tets, name, l1).values.numpy() for name in ("T", "P"))
    assert_close(values("T * T + P", tets, at=l1), t * t + p)
    h1 = td.H1(tets, order=2)
    tets.fields["w"] = td.project(tets, "T", h1)
    assert_close(values("w * 3", tets, at=h1), tets.fields["w"].values.numpy() * 3)


def test_cell_data_into_h1_on_tetrahedra_is_refused(backend):
    tets = vtk_to_dataset(tetrahedra())
    rng = np.random.default_rng(2)
    tets.fields["C"] = td.Field(td.Constant(tets), tack.array(rng.random(tets.num_cells, dtype=np.float32)))
    with pytest.raises(fx.ExprError):
        fx.compile_expression("C + 1", tets, at=td.H1(tets, order=2))


def test_polyhedra(backend):
    points, cells = grids()
    poly = vtk_to_dataset(points, polyhedral=True)
    cpoly = with_position_flux(vtk_to_dataset(cells, polyhedral=True))
    assert_close(values("T * 2 + P", poly, at="points"), calc(points, "T * 2 + P"))
    assert_close(values("cells(T)", poly), array(vtkPointDataToCellData()(points), "T", cells=True))
    assert_close(values("T + P * 2", cpoly), calc(cells, "T + P * 2", cells=True))
    assert_close(values("fsum(flux)", cpoly), td.net_flux(cpoly, "flux").values.numpy())


def test_float64(f64_backend):
    points, _ = grids()
    data = vtk_to_dataset(points, dtype=tack.f64)
    for name in ("T", "P", "V"):
        f = data.fields[name]
        data.fields[name] = td.Field(f.space, f.values.astype(tack.f64))
    expr = "T % 0.3 + sqrt(P) * 1.1 + mag(V)"
    got = values(expr, data, at="points")
    assert got.dtype == np.float64
    assert_close(got, calc(points, expr), tol=1e-12)


def test_invalid_values_are_replaced_as_vtk_replaces_them(backend):
    points, _ = grids()
    assert_close(values("ln(T - 0.5)", vtk_to_dataset(points), at="points", replace_invalid=-1.0),
                 calc(points, "ln(T - 0.5)", replace=-1.0))
