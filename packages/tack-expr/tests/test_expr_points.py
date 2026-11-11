# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""points() of cell data, averaged inside the kernel on every mesh kind, against td.project
and VTK."""

import numpy as np
import pytest
from _expr_checks import assert_close, values, vector_array

import tack
import tack.dataset as td

pytestmark = pytest.mark.quick


def uniform(_):
    return td.uniform_grid((13, 9, 7), spacing=(0.1, 0.2, 0.3))


def planar(_):
    return td.uniform_grid((17, 11, 1))


def rectilinear(rng):
    return td.rectilinear_grid(*(np.cumsum(rng.random(n)) for n in (9, 6, 5)))


def axes_renumbered(_):
    return td.uniform_grid((1, 9, 7))


def _vtk_grid():
    pytest.importorskip("vtkmodules.vtkFiltersCore")
    from vtkmodules.vtkCommonDataModel import vtkImageData

    return vtkImageData(dimensions=(11, 8, 6), spacing=(0.1, 0.1, 0.1))


def polyhedra(_):
    from vtkmodules.vtkFiltersCore import vtkAppendFilter

    from tack.interop.vtk import vtk_to_dataset

    return vtk_to_dataset(vtkAppendFilter()(_vtk_grid()), polyhedral=True)


def mixed_vtk():
    """Half the voxels of a grid, the other half cut into tetrahedra, sharing points."""
    img = _vtk_grid()
    from vtkmodules.vtkCommonCore import vtkIdList
    from vtkmodules.vtkFiltersCore import vtkAppendFilter
    from vtkmodules.vtkFiltersGeneral import vtkDataSetTriangleFilter

    try:
        from vtkmodules.vtkFiltersCore import vtkExtractCells
    except ImportError:
        from vtkmodules.vtkFiltersExtraction import vtkExtractCells

    half = img.number_of_cells // 2
    parts = []
    for lo, hi in ((0, half), (half, img.number_of_cells)):
        ids = vtkIdList()
        for i in range(lo, hi):
            ids.InsertNextId(i)
        extract = vtkExtractCells()
        extract.SetCellList(ids)
        parts.append(extract(img))
    append = vtkAppendFilter(merge_points=True)
    append.AddInputData(parts[0])
    append.AddInputData(vtkDataSetTriangleFilter()(parts[1]))
    append.Update()
    return append.output


def mixed(_):
    from tack.interop.vtk import vtk_to_dataset

    data = vtk_to_dataset(mixed_vtk())
    assert len({g.shape for g in data.topology.groups() if g.count}) == 2
    return data


@pytest.mark.parametrize("make", [uniform, planar, rectilinear, axes_renumbered, polyhedra, mixed])
def test_points_of_cell_data_match_td_project_without_calling_it(backend, monkeypatch, make):
    rng = np.random.default_rng(5)
    data = make(rng)
    n = data.num_cells
    data.fields["C"] = td.Field(td.Constant(data), tack.array(rng.random(n, dtype=np.float32)))
    data.fields["W"] = td.Field(td.Constant(data), vector_array(rng.random((n, 3), dtype=np.float32)))
    project = td.project
    calls = []
    monkeypatch.setattr(td, "project", lambda *a, **k: calls.append(a) or project(*a, **k))
    got = values("points(C) * 2 + 1", data, at="points")
    gotv = values("points(W)", data, at="points")
    assert calls == []
    assert_close(got, project(data, "C", td.H1(data)).values.numpy() * 2 + 1, tol=1e-5)
    assert_close(gotv, project(data, "W", td.H1(data)).values.numpy(), tol=1e-5)


def test_mixed_mesh_matches_vtk_cell_data_to_point_data(backend):
    grid = mixed_vtk()
    from _expr_vtk import array
    from vtkmodules.vtkFiltersCore import vtkCellDataToPointData

    from tack.interop.vtk import vtk_to_dataset

    rng = np.random.default_rng(6)
    grid.cell_data["C"] = rng.random(grid.number_of_cells, dtype=np.float32)
    ref = array(vtkCellDataToPointData()(grid), "C")
    assert_close(values("points(C)", vtk_to_dataset(grid), at="points"), ref, tol=1e-5)
