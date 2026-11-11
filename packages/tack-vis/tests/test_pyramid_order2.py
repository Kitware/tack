# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Order 2 on pyramids: VTK's 18-node "C" pyramid (shapes.Pyramid's quadratic
functions), as L2 fields."""

import numpy as np
import pytest

import tack
import tack.dataset as td
from tack.dataset import shapes

pytestmark = pytest.mark.quick


@pytest.fixture(autouse=True)
def _f64(f64_backend):
    """Compared in f64, on every backend with it."""


@tack.func
def _quadratic(x):
    return 1.0 + x[0] * x[0] + 2.0 * x[1] * x[2] + 0.5 * x[2] - 0.7 * x[0] * x[1]


def _quadratic_at(x):
    return 1.0 + x[:, 0] ** 2 + 2 * x[:, 1] * x[:, 2] + 0.5 * x[:, 2] - 0.7 * x[:, 0] * x[:, 1]


def _gradient_at(x):
    return np.stack([2 * x[:, 0] - 0.7 * x[:, 1], 2 * x[:, 2] - 0.7 * x[:, 0],
                     2 * x[:, 1] + 0.5], axis=1)


@tack.func
def _linear(x):
    return 0.5 + x[0] - 2.0 * x[1] + 0.25 * x[2]


@tack.func
def _where(x):
    return [x[0], x[1], x[2]]


def _mesh(kind, n=3):
    """The unit cube in n^3 hexahedra: each split into six pyramids about its centre
    ("pyramids"), or, cell by cell, kept, split into pyramids or into wedges
    ("mixed"; the wedges' triangles meet whole quads, so it does not conform), or
    kept or split into pyramids ("conforming")."""
    h = 1.0 / n
    points = [[i * h, j * h, k * h] for k in range(n + 1) for j in range(n + 1)
              for i in range(n + 1)]
    pid = lambda i, j, k: i + (n + 1) * (j + (n + 1) * k)
    faces = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
    cells = []
    for k in range(n):
        for j in range(n):
            for i in range(n):
                h8 = [pid(i, j, k), pid(i + 1, j, k), pid(i + 1, j + 1, k), pid(i, j + 1, k),
                      pid(i, j, k + 1), pid(i + 1, j, k + 1), pid(i + 1, j + 1, k + 1),
                      pid(i, j + 1, k + 1)]
                pattern = {"pyramids": ("pyramids",), "mixed": ("hex", "pyramids", "wedges"),
                           "conforming": ("hex", "pyramids")}[kind]
                which = pattern[(i + j + k) % len(pattern)]
                if which == "hex":
                    cells.append((shapes.HEXAHEDRON, h8))
                elif which == "wedges":
                    cells += [(shapes.WEDGE, [h8[a] for a in (0, 1, 3, 4, 5, 7)]),
                              (shapes.WEDGE, [h8[a] for a in (1, 2, 3, 5, 6, 7)])]
                else:
                    points.append(list(np.mean([points[p] for p in h8], axis=0)))
                    # Each face of the hexahedron, wound outward, is a base wound
                    # away from the centre: reversed, it goes round toward the apex.
                    cells += [(shapes.PYRAMID, [h8[f] for f in face[::-1]] + [len(points) - 1])
                              for face in faces]
    types = np.array([int(t) for t, _ in cells], np.uint8)
    offsets = np.concatenate([[0], np.cumsum([len(c) for _, c in cells])])
    topology = td.UnstructuredTopology(types, offsets, np.concatenate([c for _, c in cells]),
                                       num_points=len(points))
    return td.DataSet(topology, np.array(points), dtype=tack.f64)


@pytest.mark.parametrize("kind", ["pyramids", "mixed"])
def test_l2_of_order_2_holds_every_quadratic(backend, kind):
    """VTK's order-2 pyramid holds every quadratic of the world on straight pyramids:
    one interpolated into L2 of order 2 is itself everywhere -- probed, integrated,
    and its gradient at every node, the apex's (extrapolated along the axis, where
    the Jacobian is singular) too."""
    data = _mesh(kind)
    space = td.L2(data, order=2)
    data.fields["u"] = td.interpolate(data, _quadratic, space, dtype=tack.f64)
    if kind == "pyramids":
        assert space.size == 18 * data.num_cells
    points = np.random.default_rng(4).uniform(0.02, 0.98, (200, 3))
    probed = td.probe(data, points).fields["u"].values.numpy()
    np.testing.assert_allclose(probed, _quadratic_at(points), rtol=0, atol=1e-12)
    total = td.integrate(data, "u").values["u"]
    np.testing.assert_allclose(total, 1 + 1 / 3 + 2 / 4 + 0.25 - 0.7 / 4, rtol=0, atol=1e-12)
    gradients = td.gradients(data, "u", at=space)
    assert gradients.space is space
    nodes = td.interpolate(data, _where, space, dtype=tack.vec3(tack.f64)).values.numpy()
    np.testing.assert_allclose(gradients.values.numpy().reshape(-1, 3), _gradient_at(nodes),
                               rtol=0, atol=1e-10)


def test_bounds_hold_the_field(backend):
    """value_range bounds the field over each pyramid (from its Bernstein
    coefficients), and flatness is zero for a field the corners interpolate."""
    data = _mesh("pyramids", 2)
    space = td.L2(data, order=2)
    data.fields["u"] = td.interpolate(data, _quadratic, space, dtype=tack.f64)
    data.fields["w"] = td.interpolate(data, _linear, space, dtype=tack.f64)
    lo, hi, flat_u, flat_w = (tack.empty((data.num_cells,), tack.f64) for _ in range(4))
    for group in data.launch_groups("cells", [data.fields["u"]]):
        _bounds(data.domain_view("cells", group), data.fields["u"].view(group),
                data.fields["w"].view(group), lo, hi, flat_u, flat_w)
    pcs = tack.empty((64,), tack.vec3(tack.f64))
    tack.copyto(pcs, np.random.default_rng(6).uniform(0, 1, (64, 3)))
    samples = tack.empty((data.num_cells * 64,), tack.f64)
    for group in data.launch_groups("cells", [data.fields["u"]]):
        _samples(data.domain_view("cells", group), data.fields["u"].view(group), pcs, samples)
    samples = samples.numpy().reshape(-1, 64)
    assert (samples.min(axis=1) >= lo.numpy() - 1e-12).all()
    assert (samples.max(axis=1) <= hi.numpy() + 1e-12).all()
    assert flat_u.numpy().max() > 1e-3
    np.testing.assert_allclose(flat_w.numpy(), 0, atol=1e-12)


@tack.kernel
def _bounds(cells, u, w, lo, hi, flat_u, flat_w):
    for c in cells:
        e = cells.entity_id(c)
        lo[e], hi[e] = u.value_range(c)
        flat_u[e] = u.flatness(c)
        flat_w[e] = w.flatness(c)


@tack.kernel
def _samples(cells, u, pcs, out):
    for c in cells:
        for k in range(pcs.shape[0]):
            out[cells.entity_id(c) * pcs.shape[0] + k] = u.value(c, pcs[k])


def test_whole_cells_keep_their_values(backend):
    """A filter keeping whole cells keeps an order-2 pyramid field as it is."""
    data = _mesh("mixed")
    data.fields["u"] = td.interpolate(data, _quadratic, td.L2(data, order=2), dtype=tack.f64)
    kept = td.extract_cells(data, np.arange(0, data.num_cells, 2))
    assert kept.fields["u"].space == td.L2(kept, order=2)
    points = td.cell_centers(kept).values.numpy()
    np.testing.assert_allclose(td.probe(kept, points).fields["u"].values.numpy(),
                               _quadratic_at(points), rtol=0, atol=1e-12)


@tack.func
def _bend(x):
    return [x[0] + 0.05 * x[1] * x[1], x[1] + 0.04 * x[0] * x[2], x[2] + 0.03 * x[0] * x[0]]


@pytest.mark.parametrize("kind", ["pyramids", "mixed"])
def test_h1_of_order_2_holds_every_quadratic(backend, kind):
    """H1 of order 2 on pyramids, its values shared by entity (a value at each triangle
    face's centre where a pyramid has the face): a quadratic is itself everywhere."""
    data = _mesh(kind)
    space = td.H1(data, order=2)
    data.fields["u"] = td.interpolate(data, _quadratic, space, dtype=tack.f64)
    points = np.random.default_rng(4).uniform(0.02, 0.98, (200, 3))
    np.testing.assert_allclose(td.probe(data, points).fields["u"].values.numpy(),
                               _quadratic_at(points), rtol=0, atol=1e-12)
    np.testing.assert_allclose(td.integrate(data, "u").values["u"],
                               1 + 1 / 3 + 2 / 4 + 0.25 - 0.7 / 4, rtol=0, atol=1e-12)
    centers = td.cell_centers(data).values.numpy()
    np.testing.assert_allclose(td.gradients(data, "u", at="cells").values.numpy().reshape(-1, 3),
                               _gradient_at(centers), rtol=0, atol=1e-10)
    assert (td.cell_geometry(data).measure.values.numpy() > 0).all()


@pytest.mark.parametrize("kind", ["pyramids", "mixed"])
def test_a_curved_geometry_of_order_2(backend, kind):
    """A quadratic map of the cube held exactly by an order-2 geometry: the cells'
    volumes sum to its volume, and points are located in curved pyramids -- a field
    equal to the geometry probes back the points asked for."""
    data = _mesh(kind)
    data.geometry = td.interpolate(data, _bend, td.H1(data, order=2), dtype=tack.vec3(tack.f64))
    volume = td.cell_geometry(data).measure.values.numpy()
    assert (volume > 0).all()
    # det of the bend's Jacobian is 1 - 0.004 y z + 0.00024 x^2 y.
    np.testing.assert_allclose(volume.sum(), 1 - 0.004 / 4 + 0.00024 / 6, rtol=0, atol=1e-12)
    data.fields["x"] = td.interpolate(data, _where, td.H1(data, order=2), dtype=tack.vec3(tack.f64))
    reference = np.random.default_rng(8).uniform(0.05, 0.95, (300, 3))
    x, y, z = reference.T
    points = np.stack([x + 0.05 * y * y, y + 0.04 * x * z, z + 0.03 * x * x], axis=1)
    probed = td.probe(data, points).fields["x"].values.numpy()
    np.testing.assert_allclose(probed, points, rtol=0, atol=1e-9)


@tack.func
def _ball(x):
    return ((x[0] - 0.5) * (x[0] - 0.5) + (x[1] - 0.5) * (x[1] - 0.5)
            + (x[2] - 0.5) * (x[2] - 0.5))


@pytest.mark.parametrize("kind", ["pyramids", "conforming"])
@pytest.mark.parametrize("space", ["H1", "L2"])
def test_cuts_of_order_2_pyramids(backend, kind, space):
    """Cuts subdivide pyramids at level 2 into 6 pyramids and 4 tetrahedra on their
    order-2 nodes: a closed level set's contour is watertight where pyramids meet
    pyramids and hexahedra, and near the level; a clip's two sides fill the cube;
    threshold keeps whole pyramids and cuts the rest."""
    data = _mesh(kind)
    data.fields["u"] = td.interpolate(data, _ball, getattr(td, space)(data, order=2),
                                      dtype=tack.f64)
    surface = td.contour(data, "u", [0.09])
    assert surface.num_cells > 100
    open_edges = td.feature_edges(surface, boundary=True, feature=False, non_manifold=True,
                                  manifold=False)
    assert open_edges.num_cells == 0
    x = surface.positions()
    radius = np.linalg.norm(x - 0.5, axis=1)
    assert np.abs(radius - 0.3).max() < 0.03
    inside = sum(td.integrate(p).measure for p in td.clip(data, "u", 0.09, inside=True))
    outside = sum(td.integrate(p).measure for p in td.clip(data, "u", 0.09))
    np.testing.assert_allclose(inside + outside, 1.0, rtol=0, atol=1e-12)
    assert 0.08 < inside < 4 / 3 * np.pi * 0.3 ** 3
    whole, cut = td.threshold(data, "u", -1.0, 0.15)
    assert whole.num_cells and cut.num_cells
    made = {int(shapes.PYRAMID), int(shapes.TETRA), int(shapes.HEXAHEDRON)}
    assert set(cut.topology.types.numpy().tolist()) <= made


def test_what_pyramids_refuse(backend):
    data = _mesh("pyramids", 1)
    with pytest.raises(NotImplementedError, match="order 2 on pyramids"):
        _ = td.L2(data, order=3).size
    with pytest.raises(NotImplementedError, match="order 2 on pyramids"):
        _ = td.L2(data, order=2, basis="gauss_legendre").size
    with pytest.raises(ValueError, match="stored in blocks"):
        _ = td.L2(data, order=3, layout="cells").layout
    with pytest.raises(NotImplementedError, match="H1 order 3 on Pyramid"):
        _ = td.H1(data, order=3).size
    data.fields["u"] = td.interpolate(data, _quadratic, td.L2(data, order=2), dtype=tack.f64)
    with pytest.raises(NotImplementedError, match="level 1 or 2"):
        td.subdivide(data, 3)
