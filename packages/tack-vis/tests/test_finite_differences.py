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


# ── Hidden points and cells ─────────────────────────────────────────

def _blocked(marks, sizes):
    """Where some axis of more than one entity has no visible neighbour on either
    side: there VTK takes the world axis of that number as the grid's direction
    and Tack a normal to the others, which differ off planes of world axes."""
    hidden = (marks != 0).reshape(sizes[::-1])
    blocked = np.zeros_like(hidden)
    for axis in range(3):
        if hidden.shape[axis] < 2:
            continue
        gone = np.ones_like(hidden)
        before = gone.copy()
        after = gone.copy()
        index = [slice(None)] * 3
        index[axis] = slice(1, None)
        shifted = [slice(None)] * 3
        shifted[axis] = slice(None, -1)
        before[tuple(index)] = hidden[tuple(shifted)]
        after[tuple(shifted)] = hidden[tuple(index)]
        blocked |= before & after
    return blocked.reshape(-1)


def _hide(grid, on, fraction, seed=5):
    """Marks hiding about ``fraction`` of ``grid``'s points (cells), as a VTK ghost
    array: the marks."""
    n = grid.number_of_points if on == "points" else grid.number_of_cells
    marks = np.zeros(n, np.uint8)
    bit = td.Ghosts.HIDDENPOINT if on == "points" else td.Ghosts.HIDDENCELL
    marks[np.random.default_rng(seed).random(n) < fraction] = bit
    (grid.point_data if on == "points" else grid.cell_data)["vtkGhostType"] = (
        vtk.vtkUnsignedCharArray(marks))
    return marks


@needs_vtk
@pytest.mark.parametrize("kind", ["image", "rectilinear", "curvilinear"])
@pytest.mark.parametrize("dims", [(9, 8, 7), (9, 1, 7)])
@pytest.mark.parametrize("on", ["points", "cells"])
def test_order_2_around_hidden_values_is_vtks_gradient(kind, dims, on):
    """A hidden neighbour makes VTK's stencil one-sided, and two leave no derivative
    along the axis; VTK writes 0 at the hidden points themselves, Tack NaN."""
    grid = _vtk_grid(kind, dims)
    if on == "points":
        x = np.array([grid.GetPoint(p) for p in range(grid.number_of_points)])
        attributes = grid.point_data
        sizes = dims
    else:
        centers = vtk.vtkCellCenters()(grid)
        x = np.array([centers.GetPoint(c) for c in range(centers.number_of_points)])
        attributes = grid.cell_data
        sizes = tuple(max(n - 1, 1) for n in dims)
    s, v = _sampled(x)
    attributes["s"] = vtk.vtkDoubleArray(s)
    attributes["v"] = vtk.vtkDoubleArray(v)
    marks = _hide(grid, on, 0.2)
    data = vtk_to_dataset(grid, dtype=tack.f64)
    hidden = marks != 0
    keep = ~hidden & (~_blocked(marks, sizes) if kind == "curvilinear" else True)
    assert hidden.any() and keep.sum() > len(keep) // 2
    for name in ("s", "v"):
        ours = td.finite_differences(data, name).values.numpy()
        theirs = _vtk_gradient(grid, name, on).reshape(ours.shape)
        assert np.isnan(ours[hidden]).all() and not np.isnan(ours[~hidden]).any()
        np.testing.assert_allclose(ours[keep], theirs[keep], rtol=1e-11, atol=1e-11,
                                   err_msg=name)


def _marked(data, fraction, on="points", bit=None, seed=6):
    n = data.num_points if on == "points" else data.num_cells
    marks = np.zeros(n, np.uint8)
    if bit is None:
        bit = td.Ghosts.HIDDENPOINT if on == "points" else td.Ghosts.HIDDENCELL
    marks[np.random.default_rng(seed).random(n) < fraction] = bit
    array = tack.array(marks, tack.u8)
    data.ghosts = td.Ghosts(points=array) if on == "points" else td.Ghosts(cells=array)
    return marks


@pytest.mark.parametrize("order", [2, 4, 6])
@pytest.mark.parametrize("derivative", [1, 2])
@pytest.mark.parametrize("on", ["points", "cells"])
def test_no_stencil_reads_a_hidden_value(order, derivative, on):
    data = _curvilinear((10, 9, 8))
    n = data.num_points if on == "points" else data.num_cells
    marks = _marked(data, 0.15, on)
    hidden = marks != 0
    values = np.random.default_rng(1).random(n)
    space = td.H1(data) if on == "points" else td.Constant(data)
    results = []
    for garbage in (1e30, -7.0):
        u = np.where(hidden, garbage, values)
        data.fields["u"] = td.Field(space, tack.array(u, tack.f64))
        results.append(td.finite_differences(data, "u", order=order,
                                             derivative=derivative).values.numpy())
    assert np.isnan(results[0][hidden]).all()
    np.testing.assert_array_equal(results[0][~hidden], results[1][~hidden])
    assert np.isfinite(results[0][~hidden]).all()


@pytest.mark.parametrize("order", [2, 4, 6])
def test_linear_fields_stay_exact_around_hidden_points(order):
    data = _curvilinear((10, 9, 8))
    marks = _marked(data, 0.1)
    a = np.array([0.7, -1.3, 2.1])
    _with(data, _points(data) @ a)
    keep = (marks == 0) & ~_blocked(marks, (10, 9, 8))
    g = td.finite_differences(data, "u", order=order).values.numpy()
    h = td.finite_differences(data, "u", order=order, derivative=2).values.numpy()
    np.testing.assert_allclose(g[keep], np.broadcast_to(a, g[keep].shape), rtol=0, atol=1e-10)
    np.testing.assert_allclose(h[keep], 0.0, rtol=0, atol=1e-8)


def test_marks_that_hide_nothing_change_nothing():
    """Duplicates (a ghost layer) and an AMR box's refined cells hold values."""
    for on, bit in (("points", td.Ghosts.DUPLICATEPOINT), ("cells", td.Ghosts.REFINEDCELL),
                    ("cells", td.Ghosts.DUPLICATECELL)):
        data = _curvilinear((8, 7, 6))
        n = data.num_points if on == "points" else data.num_cells
        space = td.H1(data) if on == "points" else td.Constant(data)
        data.fields["u"] = td.Field(space, tack.array(np.random.default_rng(2).random(n)))
        plain = td.finite_differences(data, "u", order=4, derivative=2).values.numpy()
        _marked(data, 0.3, on, bit)
        np.testing.assert_array_equal(
            td.finite_differences(data, "u", order=4, derivative=2).values.numpy(), plain)


@pytest.mark.parametrize("derivative", [1, 2])
@pytest.mark.parametrize("layout", ["aos", "soa"])
def test_a_vector_field_in_either_layout(derivative, layout):
    """The positions as a field: the identity for gradient, zero for second
    derivatives, whether the components are interleaved or arrays of their own."""
    data = td.uniform_grid((5, 4, 3), dtype=tack.f64)
    data.fields["v"] = td.Field(td.H1(data), tack.array(data.positions(), tack.vec3(tack.f64),
                                                        layout=layout))
    got = td.finite_differences(data, "v", derivative=derivative).values.numpy()
    want = np.eye(3).reshape(-1) if derivative == 1 else np.zeros(27)
    np.testing.assert_allclose(got, np.broadcast_to(want, got.shape), atol=1e-12)


@pytest.mark.parametrize("dims,spacing", [((4,), (0.0,)), ((4, 3), (0.0, 1.0))])
@pytest.mark.parametrize("derivative", [1, 2])
def test_a_collapsed_grid_has_zero_derivatives_not_nan(dims, spacing, derivative):
    """A line of zero length, or a plane collapsed along one axis: the positions'
    derivatives are singular, so the derivatives are zero, as documented."""
    data = td.uniform_grid(dims, spacing=spacing, dtype=tack.f64)
    data.fields["u"] = td.Field(td.H1(data), tack.array(np.arange(data.num_points,
                                                                  dtype=np.float64)))
    got = td.finite_differences(data, "u", derivative=derivative).values.numpy()
    np.testing.assert_array_equal(got, 0.0)
