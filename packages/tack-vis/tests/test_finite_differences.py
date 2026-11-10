# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""``finite_differences`` on images, rectilinear and curvilinear grids (#315): order 2
against VTK's vtkGradientFilter, which takes structured data by central
differences in index space; linear fields exact at every order on any grid;
polynomials up to the order exact on images; and the error falling at the
order's rate on a smoothly curved grid, for the gradient and the second
derivatives alike."""

import numpy as np
import pytest

import tack
import tack.dataset as td
from tack.dataset import units

try:
    import vtkmodules.all as vtk

    from tack.interop.vtk import vtk_to_dataset
except ImportError:
    vtk = None

needs_vtk = pytest.mark.skipif(vtk is None, reason="needs VTK, the reference")

pytestmark = pytest.mark.quick      # the inner-loop set; see CLAUDE.md, Tests


@pytest.fixture(autouse=True)
def _every_device_with_f64(f64_backend):
    """Compared in f64: on every device that has it."""


def _curved(dims, scale=1.0):
    """The positions of a smoothly curved grid of ``dims`` points, x fastest; a grid of
    fewer dimensions lies in a plane of world axes (on the x axis), curved in it."""
    axes = [np.linspace(0.0, 1.0, n) if n > 1 else np.zeros(1) for n in dims]
    i, j, k = np.meshgrid(*axes, indexing="ij")
    x = i + 0.1 * np.sin(2 * (j + k))
    y = (j + 0.3 * j ** 2 + 0.1 * i * k) * (dims[1] > 1)
    z = (k + 0.05 * np.sin(3 * i)) * (dims[2] > 1)
    return scale * np.stack([x, y, z], -1).transpose(2, 1, 0, 3).reshape(-1, 3)


def _vtk_grid(kind, dims):
    rng = np.random.default_rng(3)
    if kind == "image":
        return vtk.vtkImageData(dimensions=dims, spacing=(0.3, 0.25, 0.2),
                                origin=(0.1, -0.2, 0.3))
    if kind == "rectilinear":
        grid = vtk.vtkRectilinearGrid(dimensions=dims)
        for axis, n in enumerate(dims):
            coordinates = vtk.vtkDoubleArray(np.cumsum(rng.uniform(0.1, 0.3, n)))
            (grid.SetXCoordinates, grid.SetYCoordinates, grid.SetZCoordinates)[axis](
                coordinates)
        return grid
    grid = vtk.vtkStructuredGrid()
    grid.SetDimensions(*dims)
    grid.points = vtk.vtkPoints(data=_curved(dims, 2.0))
    return grid


def _vtk_gradient(grid, name, on):
    g = vtk.vtkGradientFilter(result_array_name="grad")
    g.SetInputArrayToProcess(0, 0, 0, 0 if on == "points" else 1, name)
    out = g(grid)
    return np.asarray((out.point_data if on == "points" else out.cell_data)["grad"])


def _sampled(x):
    s = np.sin(x[:, 0]) * x[:, 1] + x[:, 2] ** 2 + 0.5 * x[:, 0] * x[:, 2]
    v = np.c_[np.sin(x[:, 1]), x[:, 0] * x[:, 2], x[:, 1] ** 2 - x[:, 0]]
    return s, v


@needs_vtk
@pytest.mark.parametrize("kind", ["image", "rectilinear", "curvilinear"])
@pytest.mark.parametrize("dims", [(9, 8, 7), (9, 1, 7), (6, 7, 1), (10, 1, 1)])
@pytest.mark.parametrize("on", ["points", "cells"])
def test_order_2_is_vtks_gradient(kind, dims, on):
    if on == "cells" and sum(n > 2 for n in dims) == 0:
        pytest.skip("one cell")
    grid = _vtk_grid(kind, dims)
    if on == "points":
        x = np.array([grid.GetPoint(p) for p in range(grid.number_of_points)])
        attributes = grid.point_data
    else:
        centers = vtk.vtkCellCenters()(grid)
        x = np.array([centers.GetPoint(c) for c in range(centers.number_of_points)])
        attributes = grid.cell_data
    s, v = _sampled(x)
    attributes["s"] = vtk.vtkDoubleArray(s)
    attributes["v"] = vtk.vtkDoubleArray(v)
    data = vtk_to_dataset(grid, dtype=tack.f64)
    for name in ("s", "v"):
        ours = td.finite_differences(data, name).values.numpy()
        theirs = _vtk_gradient(grid, name, on).reshape(ours.shape)
        np.testing.assert_allclose(ours, theirs, rtol=1e-11, atol=1e-11, err_msg=name)


def _image(dims, dtype=tack.f64):
    return td.uniform_grid(dims, spacing=(0.3, 0.25, 0.2), origin=(0.1, -0.2, 0.3),
                           dtype=dtype)


def _curvilinear(dims, dtype=tack.f64):
    from tack.dataset.dataset import grid_topology

    topology = grid_topology(dims)
    x = _curved(dims)
    geometry = td.Field(td.H1(topology), tack.array(x.astype(dtype.numpy_dtype),
                                                    tack.vec(3, dtype)))
    return td.DataSet(topology, geometry)


def _with(data, values):
    data.fields["u"] = td.Field(td.H1(data), tack.array(values, tack.f64))
    return data


def _points(data):
    return data.geometry.values.numpy().astype(np.float64)


@pytest.mark.parametrize("dims", [(8, 7, 6), (8, 1, 6), (9, 1, 1)])
@pytest.mark.parametrize("order", [2, 4, 6])
def test_linear_fields_are_exact_on_any_grid(dims, order):
    a = np.array([0.7, -1.3, 2.1])
    for data in (_image(dims), _curvilinear(dims)):
        x = _points(data)
        _with(data, x @ a)
        g = td.finite_differences(data, "u", order=order).values.numpy()
        h = td.finite_differences(data, "u", order=order, derivative=2).values.numpy()
        # A grid of fewer dimensions has the gradient's part in its plane (on its line).
        expected = np.broadcast_to(_in_plane(data, a), g.shape)
        np.testing.assert_allclose(g, expected, rtol=0, atol=1e-11)
        np.testing.assert_allclose(h, 0.0, rtol=0, atol=1e-9)


def _in_plane(data, a):
    """``a`` without its parts along the directions the grid does not span: its
    plane or line is one of world axes (``_curved``), spanned by its first steps."""
    x = _points(data)
    dims = data.topology.point_dims
    steps = []
    stride = 1
    for n in dims:
        steps.append(x[stride] - x[0])
        stride *= n
    basis, _ = np.linalg.qr(np.array(steps).T)
    return basis @ (basis.T @ a)


@pytest.mark.parametrize("order", [2, 4, 6])
def test_polynomials_up_to_the_order_are_exact_on_images(order):
    data = _image((11, 10, 9))
    x = _points(data)
    p = order if order > 2 else 1        # order 2 is first order at the edges
    u = x[:, 0] ** p + x[:, 1] ** (p - 1) * x[:, 2] + x[:, 0] * x[:, 1] * x[:, 2] ** (p - 1)
    _with(data, u)
    g = td.finite_differences(data, "u", order=order).values.numpy()
    expected = np.c_[p * x[:, 0] ** (p - 1) + x[:, 1] * x[:, 2] ** (p - 1),
                     (p - 1) * x[:, 1] ** max(p - 2, 0) * x[:, 2] * (p > 1)
                     + x[:, 0] * x[:, 2] ** (p - 1),
                     x[:, 1] ** (p - 1) + (p - 1) * x[:, 0] * x[:, 1] * x[:, 2] ** max(p - 2, 0)]
    np.testing.assert_allclose(g, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("order", [2, 4, 6])
def test_second_derivatives_of_polynomials_are_exact_on_images(order):
    data = _image((11, 10, 9))
    x = _points(data)
    p = order if order > 2 else 2        # order 2's edges: three points, quadratics
    u = x[:, 0] ** p + x[:, 0] * x[:, 1] + x[:, 2] ** p * (order > 2) + x[:, 1] * x[:, 2]
    _with(data, u)
    h = td.finite_differences(data, "u", order=order, derivative=2).values.numpy()
    z = np.zeros(len(x))
    expected = np.c_[p * (p - 1) * x[:, 0] ** (p - 2), z + 1, z,
                     z + 1, z, z + 1,
                     z, z + 1, p * (p - 1) * x[:, 2] ** (p - 2) * (order > 2)]
    np.testing.assert_allclose(h, expected, rtol=1e-9, atol=1e-8)


def _smooth(x):
    u = np.sin(x[:, 0]) * np.cos(1.3 * x[:, 1]) + np.exp(0.5 * x[:, 2])
    s, c = np.sin(x[:, 0]), np.cos(x[:, 0])
    s1, c1 = np.sin(1.3 * x[:, 1]), np.cos(1.3 * x[:, 1])
    z = np.zeros(len(x))
    g = np.c_[c * c1, -1.3 * s * s1, 0.5 * np.exp(0.5 * x[:, 2])]
    h = np.c_[-s * c1, -1.3 * c * s1, z, -1.3 * c * s1, -1.69 * s * c1, z,
              z, z, 0.25 * np.exp(0.5 * x[:, 2])]
    return u, g, h


@pytest.mark.parametrize("order", [2, 4, 6])
def test_the_error_falls_at_the_orders_rate_on_a_curved_grid(order):
    errors = []
    for n in (14, 28):
        data = _curvilinear((n, n, n))
        x = _points(data)
        u, g, h = _smooth(x)
        _with(data, u)
        inside = np.zeros((n, n, n), bool)
        inside[3:-3, 3:-3, 3:-3] = True
        # Order 2's edges are first order, as VTK's: its rate is the interior's.
        where = inside.reshape(-1) if order == 2 else slice(None)
        dg = td.finite_differences(data, "u", order=order).values.numpy() - g
        dh = td.finite_differences(data, "u", order=order, derivative=2).values.numpy() - h
        errors.append((np.abs(dg[where]).max(), np.abs(dh[where]).max()))
    rates = np.log2(np.array(errors[0]) / np.array(errors[1]))
    assert (rates > order - 0.5).all(), rates


def test_vectors_give_a_row_per_component_and_flow_quantities_take_them():
    data = _curvilinear((9, 8, 7))
    x = _points(data)
    v = np.c_[np.sin(x[:, 1]), x[:, 0] * x[:, 2], x[:, 1] ** 2 - x[:, 0]]
    data.fields["v"] = td.Field(td.H1(data), tack.array(v, tack.vec(3, tack.f64)))
    whole = td.finite_differences(data, "v", order=4).values.numpy()
    hessians = td.finite_differences(data, "v", order=4, derivative=2).values.numpy()
    assert whole.shape == (len(x), 9) and hessians.shape == (len(x), 27)
    for c in range(3):
        _with(data, v[:, c])
        np.testing.assert_array_equal(
            whole[:, 3 * c:3 * c + 3], td.finite_differences(data, "u", order=4).values.numpy())
        part = td.finite_differences(data, "u", order=4, derivative=2).values.numpy()
        np.testing.assert_array_equal(hessians[:, 9 * c:9 * c + 9], part)
        np.testing.assert_allclose(part.reshape(-1, 3, 3), part.reshape(-1, 3, 3).transpose(0, 2, 1),
                                   rtol=1e-12, atol=1e-12)
    quantities = td.flow_quantities(data, td.finite_differences(data, "v"))
    assert quantities.vorticity.values.shape[0] == len(x)


def test_units_are_per_length_and_per_length_squared():
    data = td.uniform_grid((5, 4, 3), dtype=tack.f64, unit="m")
    data.fields["t"] = td.Field(td.H1(data), tack.array(np.arange(60.0), tack.f64), "degC")
    assert td.finite_differences(data, "t").unit == units.derived("K", ("m", -1))
    assert td.finite_differences(data, "t", derivative=2).unit == units.derived(
        "K", ("m", -2))


def test_what_it_refuses():
    data = _with(_image((5, 4, 3)), np.arange(60.0))
    with pytest.raises(ValueError, match="2, 4 or 6"):
        td.finite_differences(data, "u", order=3)
    with pytest.raises(ValueError, match="1 or 2"):
        td.finite_differences(data, "u", derivative=3)
    with pytest.raises(TypeError, match="structured grid"):
        td.finite_differences(td.tetrahedralize(data), "u")
    with pytest.raises(NotImplementedError, match="point data"):
        td.finite_differences(data, td.project(data, "u", td.L2(data)))
    marks = np.zeros(60, np.uint8)
    marks[7] = td.Ghosts.HIDDENPOINT
    data.ghosts = td.Ghosts(points=tack.array(marks, tack.u8))
    with pytest.raises(NotImplementedError, match="hidden points"):
        td.finite_differences(data, "u")
