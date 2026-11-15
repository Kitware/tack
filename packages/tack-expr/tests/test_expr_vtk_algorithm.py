# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The Calculator VTK algorithm (tack.interop.vtk.algorithms): an expression in a VTK
pipeline, as vtkArrayCalculator, against tack_expr.evaluate on the same dataset."""

import numpy as np
import pytest

pytest.importorskip("vtkmodules.vtkImagingCore")

import tack_expr as fx
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkImagingCore import vtkRTAnalyticSource

from tack.interop.vtk import algorithms as ta
from tack.interop.vtk import vtk_to_dataset

pytestmark = pytest.mark.quick


def _wavelet():
    return vtkRTAnalyticSource(whole_extent=(-6, 6, -6, 6, -6, 6))()


@pytest.mark.parametrize("at, where", [("points", "point_data"), ("cells", "cell_data")])
def test_calculator_adds_the_expression_as_an_array(at, where):
    image = _wavelet()
    text = "sqrt(RTData) * 2 + coordsX"
    out = ta.Calculator(expression=text, at=at, name="r")(image)
    assert isinstance(out, vtkImageData)
    assert "r" in getattr(out, where).keys() and "r" not in getattr(image, where).keys()
    direct = fx.evaluate(text, vtk_to_dataset(image, dtype=None), at=at)
    np.testing.assert_allclose(np.asarray(getattr(out, where)["r"]), direct.values.numpy(),
                               rtol=1e-6)


def test_calculator_derivatives_follow_the_cells():
    image = _wavelet()
    out = ta.Calculator(expression="grad(RTData)", at="cells", name="g")(image)
    direct = fx.evaluate("grad(RTData)", vtk_to_dataset(image, dtype=None), at="cells")
    np.testing.assert_allclose(np.asarray(out.cell_data["g"]), direct.values.numpy(),
                               rtol=1e-6)


def test_calculator_needs_an_expression():
    algorithm = ta.Calculator()
    algorithm(_wavelet())
    assert isinstance(algorithm.error, ValueError)
