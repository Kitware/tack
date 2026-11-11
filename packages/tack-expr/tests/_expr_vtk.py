# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""VTK references for the tack_expr tests: grids with random arrays and vtkArrayCalculator."""

import functools

import numpy as np
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from vtkmodules.vtkCommonCore import VTK_DOUBLE, vtkObject
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkFiltersCore import vtkAppendFilter, vtkArrayCalculator
from vtkmodules.vtkFiltersGeneral import vtkDataSetTriangleFilter

vtkObject.GlobalWarningDisplayOff()

DIMS = (16, 12, 9)


def run_filter(filter_class, data):
    f = filter_class()
    f.SetInputData(data)
    f.Update()
    return f.GetOutput()


def _arrays(rng, n):
    return {"T": rng.random(n, dtype=np.float32), "P": rng.random(n, dtype=np.float32),
            "V": rng.random((n, 3), dtype=np.float32) * 2 - 1,
            "W": rng.random((n, 3), dtype=np.float32) * 2 - 1}


def image(point=None, cell=None):
    img = vtkImageData()
    img.SetDimensions(DIMS)
    img.SetOrigin(0.5, -1.0, 2.0)
    img.SetSpacing(0.1, 0.2, 0.05)
    for attrs, arrays in ((img.GetPointData(), point), (img.GetCellData(), cell)):
        for name, a in (arrays or {}).items():
            arr = numpy_to_vtk(a, deep=True)
            arr.SetName(name)
            attrs.AddArray(arr)
    return img


@functools.cache
def grids():
    """Unstructured grids of voxels, one with point arrays T, P, V, W and one with cell arrays."""
    rng = np.random.default_rng(1)
    img = image()
    point = image(point=_arrays(rng, img.GetNumberOfPoints()))
    cell = image(cell=_arrays(rng, img.GetNumberOfCells()))
    return run_filter(vtkAppendFilter, point), run_filter(vtkAppendFilter, cell)


@functools.cache
def tetrahedra():
    return run_filter(vtkDataSetTriangleFilter, grids()[0])


def array(data, name, cells=False):
    return vtk_to_numpy((data.GetCellData() if cells else data.GetPointData()).GetArray(name))


def calc(data, expr, cells=False, replace=None):
    c = vtkArrayCalculator()
    c.SetInputData(data)
    c.SetAttributeTypeToCellData() if cells else c.SetAttributeTypeToPointData()
    c.AddScalarArrayName("T")
    c.AddScalarArrayName("P")
    c.AddVectorArrayName("V")
    c.AddVectorArrayName("W")
    for k, comp in enumerate("XYZ"):
        c.AddScalarVariable(f"V_{comp}", "V", k)
        c.AddScalarVariable(f"W_{comp}", "W", k)
        if not cells:
            c.AddCoordinateScalarVariable(f"coords{comp}", k)
    if not cells:
        c.AddCoordinateVectorVariable("coords")
    c.SetFunction(expr)
    c.SetResultArrayName("result")
    c.SetResultArrayType(VTK_DOUBLE)
    if replace is not None:
        c.ReplaceInvalidValuesOn()
        c.SetReplacementValue(replace)
    c.Update()
    return array(c.GetOutput(), "result", cells)
