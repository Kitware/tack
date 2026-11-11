# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""VTK references for the tack_expr tests: grids with random arrays and vtkArrayCalculator."""

import functools

import numpy as np
from vtkmodules.vtkCommonCore import VTK_DOUBLE, vtkObject
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkFiltersCore import vtkAppendFilter, vtkArrayCalculator
from vtkmodules.vtkFiltersGeneral import vtkDataSetTriangleFilter

vtkObject.GlobalWarningDisplayOff()

DIMS = (16, 12, 9)


def _arrays(rng, n):
    return {"T": rng.random(n, dtype=np.float32), "P": rng.random(n, dtype=np.float32),
            "V": rng.random((n, 3), dtype=np.float32) * 2 - 1,
            "W": rng.random((n, 3), dtype=np.float32) * 2 - 1}


def image(point=None, cell=None):
    img = vtkImageData(dimensions=DIMS, origin=(0.5, -1.0, 2.0), spacing=(0.1, 0.2, 0.05))
    for attrs, arrays in ((img.point_data, point), (img.cell_data, cell)):
        for name, a in (arrays or {}).items():
            attrs[name] = a
    return img


@functools.cache
def grids():
    """Unstructured grids of voxels, one with point arrays T, P, V, W and one with cell arrays."""
    rng = np.random.default_rng(1)
    img = image()
    point = image(point=_arrays(rng, img.number_of_points))
    cell = image(cell=_arrays(rng, img.number_of_cells))
    return vtkAppendFilter()(point), vtkAppendFilter()(cell)


@functools.cache
def tetrahedra():
    return vtkDataSetTriangleFilter()(grids()[0])


def array(data, name, cells=False):
    return np.asarray((data.cell_data if cells else data.point_data)[name])


def calc(data, expr, cells=False, replace=None):
    c = vtkArrayCalculator(function=expr, result_array_name="result", result_array_type=VTK_DOUBLE)
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
    if replace is not None:
        c.replace_invalid_values = True
        c.replacement_value = replace
    return array(c(data), "result", cells)
