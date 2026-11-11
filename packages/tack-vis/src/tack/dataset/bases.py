# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Lagrange bases of any order on the tensor shapes: segments, quads, pixels,
hexahedra and voxels.

A basis is a shape, an order ``p`` and a family of 1D nodes on ``[0, 1]`` (the
shapes' parametric range):

- ``"gauss_lobatto"``: the ends and the roots of the derivative of the Legendre
  polynomial of degree ``p`` -- MFEM's default for ``H1``, Nek5000's and
  deal.II's; better conditioned than equispaced nodes at high order;
- ``"equispaced"``: ``j / p`` -- VTK's Lagrange cells, Exodus, gmsh;
- ``"gauss_legendre"``: the roots of the Legendre polynomial of degree ``p + 1``,
  inside the segment -- MFEM's default for ``L2``; for ``L2`` only, since
  shared values need nodes at the ends.

At order 2 or less the two closed families are the same nodes. A cell's
nodes are the products of the 1D nodes along its axes, numbered
lexicographically, the first axis fastest: node ``n`` is ``(i, j, k)`` with
``n = i + (p + 1) * (j + (p + 1) * k)``, at parametric coordinates
``(x[i], x[j], x[k])``. Its basis function is the product of the 1D Lagrange
polynomials, evaluated in barycentric form from the nodes and their weights
(``weights``), which kernels read as small arrays in the field's own float type.

``H1`` shares values: a node at a corner is the point's, one inside an edge,
quad face or cell belongs to that entity (``dof_tables`` classifies each node,
and the field views orient edges from their lower point id and faces in their
own row's frame). ``L2`` keeps each cell's ``(p + 1)^d`` values in node order.
"""

import functools

import numpy as np

from tack.dataset import shapes

__all__ = ["BERNSTEIN", "FAMILIES", "SERENDIPITY", "bernstein", "dof_tables", "entity_nodes", "has_bases",
           "lattice_ijk", "nodes", "simplex_bernstein", "simplex_lattice", "simplex_shape",
           "simplex_subdivision", "tensor_shape", "view_index", "view_order", "wedge_bernstein",
           "wedge_lattice", "wedge_shape", "wedge_subdivision", "weights"]

FAMILIES = ("gauss_lobatto", "equispaced", "gauss_legendre", "bernstein", "serendipity")
#: The families whose values at the ends are the polynomial's there, which shared
#: values need: the closed node sets, Bernstein's (its end coefficients) and
#: serendipity.
CLOSED = ("gauss_lobatto", "equispaced", "bernstein", "serendipity")
#: Serendipity (``H1`` of order 2 alone): the full order-2 space less its values at
#: quad faces' centres and cells' insides, which its functions determine from
#: the corners and edges (quad8, hex20, wedge15). Read as the equispaced order-2
#: basis, so its nodes, weights and Bernstein matrix are that basis'.
SERENDIPITY = "serendipity"
#: The Bernstein polynomials: coefficients, not values at nodes. Its "nodes" are
#: its Greville points (``i / p``, the equispaced nodes), where a value would be
#: interpolated before converting (``bernstein("equispaced", p)``), and its
#: "weights" the binomial coefficients ``C(p, i)`` its polynomials are scaled by.
BERNSTEIN = "bernstein"

#: Node kinds in ``dof_tables``.
CORNER, EDGE, FACE, INTERIOR = 0, 1, 2, 3


def _legendre_roots(polynomial, guesses):
    """``guesses`` polished by Newton's method on ``polynomial``, a NumPy Legendre series."""
    x = np.asarray(guesses, np.float64)
    derivative = polynomial.deriv()
    for _ in range(100):
        step = polynomial(x) / derivative(x)
        x = x - step
        if np.max(np.abs(step), initial=0.0) < 1e-16:
            break
    return x


@functools.cache
def nodes(family, order):
    """The ``order + 1`` nodes of ``family`` on ``[0, 1]``, increasing, as an f64 array,
    symmetric about ``0.5`` to the last bit."""
    if family not in FAMILIES:
        raise ValueError(f"a basis is one of {', '.join(FAMILIES)}, not {family!r}")
    if family in (BERNSTEIN, SERENDIPITY):
        return nodes("equispaced", order)
    p = int(order)
    if p < 0:
        raise ValueError(f"an order is at least 0, not {order}")
    if p == 0:
        x = np.array([0.0])                     # one node, at the middle
    elif family == "equispaced":
        x = np.linspace(-1.0, 1.0, p + 1)
    elif family == "gauss_lobatto":
        legendre = np.polynomial.legendre.Legendre.basis(p)
        inner = np.cos(np.pi * np.arange(p - 1, 0, -1) / p)       # Chebyshev-Lobatto guesses
        x = np.concatenate([[-1.0], _legendre_roots(legendre.deriv(), inner), [1.0]])
    else:
        legendre = np.polynomial.legendre.Legendre.basis(p + 1)
        x = _legendre_roots(legendre, np.polynomial.legendre.leggauss(p + 1)[0])
    x = 0.5 * (np.sort(x) + 1.0)
    half = (p + 1) // 2
    x[p - np.arange(half)] = 1.0 - x[:half]     # the upper half mirrors the lower
    if p % 2 == 0:
        x[p // 2] = 0.5
    x.setflags(write=False)
    return x


@functools.cache
def weights(family, order):
    """The barycentric weights of ``nodes(family, order)``: ``1 / prod(x_j - x_m)``
    over ``m != j``, so the Lagrange polynomial of node ``j`` at ``t`` is
    ``weights[j] * prod(t - x_m)``. Bernstein's are ``C(p, j)``: its polynomial ``j``
    is ``weights[j] * t^j (1 - t)^(p - j)``."""
    if family == BERNSTEIN:
        from math import comb

        w = np.array([comb(int(order), j) for j in range(int(order) + 1)], np.float64)
        w.setflags(write=False)
        return w
    if family == SERENDIPITY:
        return weights("equispaced", order)
    x = nodes(family, order)
    difference = x[:, None] - x[None, :]
    np.fill_diagonal(difference, 1.0)
    w = 1.0 / np.prod(difference, axis=1)
    w.setflags(write=False)
    return w


@functools.cache
def bernstein(family, order):
    """The matrix taking a 1D polynomial's values at ``nodes(family, order)`` to its
    Bernstein coefficients, row ``i`` the ``i``-th coefficient. The polynomial lies
    between its least and greatest coefficients (the convex hull property), so a
    tensor product of these bounds a cell's geometry, or a field over the cell.
    Bernstein's own is the identity."""
    from math import comb

    if family == BERNSTEIN:
        m = np.eye(int(order) + 1)
        m.setflags(write=False)
        return m
    if family == SERENDIPITY:
        return bernstein("equispaced", order)
    x = nodes(family, order)
    p = int(order)
    at = np.array([[comb(p, j) * t ** j * (1.0 - t) ** (p - j) for j in range(p + 1)]
                   for t in x])
    m = np.linalg.inv(at)
    m.setflags(write=False)
    return m


# ── Shapes: their corners, edges and faces on the host ──────────────

#: Each tensor shape's corners as integer parametric coordinates (0 or 1), in
#: its own point order, and its edge and face tables (``shapes``' constants).
_SHAPES = {
    shapes.Line: ([(0,), (1,)], (), ()),
    shapes.Quad: ([(0, 0), (1, 0), (1, 1), (0, 1)], shapes._QUAD_EDGES, ()),
    shapes.Pixel: ([(0, 0), (1, 0), (0, 1), (1, 1)], shapes._PIXEL_EDGES, ()),
    shapes.Hexahedron: ([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                         (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)],
                        shapes._HEXAHEDRON_EDGES, shapes._HEXAHEDRON_FACES),
    shapes.Voxel: ([(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
                    (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)],
                   shapes._VOXEL_EDGES, shapes._VOXEL_FACES),
}


def tensor_shape(shape):
    """Whether ``shape`` has tensor bases of any order here."""
    return shape in _SHAPES


# ── Simplices: triangles and tetrahedra ─────────────────────────────

#: Each simplex's corners as integer parametric coordinates, its edge table and
#: its faces (triangles, going round outward).
_SIMPLICES = {
    shapes.Triangle: ([(0, 0), (1, 0), (0, 1)], shapes._TRIANGLE_EDGES, ()),
    "segment": ([(0,), (1,)], (0, 1), ()),
    shapes.Tetra: ([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)], shapes._TETRA_EDGES,
                   ((0, 1, 3), (1, 2, 3), (2, 0, 3), (0, 2, 1))),
}


#: A segment as a 1D simplex, for the wedge's ``z`` factor (host only).
_SEGMENT = "segment"


def simplex_shape(shape):
    """Whether ``shape`` has simplex bases of any order here (equispaced nodes)."""
    return shape in _SIMPLICES and shape != _SEGMENT


def has_bases(shape):
    """Whether ``shape`` has bases of any order here: tensor, simplex or wedge."""
    return tensor_shape(shape) or simplex_shape(shape) or wedge_shape(shape)


# ── Wedges: a triangle times a segment ──────────────────────────────

#: The wedge's corners (its triangle at z = 0, then at z = 1), edges and faces
#: (two triangles, three quads), as ``shapes.Wedge`` numbers them.
_WEDGE = ([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1)],
          shapes._WEDGE_EDGES,
          ((0, 2, 1), (3, 4, 5), (0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5)))


def wedge_shape(shape):
    """Whether ``shape`` is the wedge, whose bases of any order here are a triangle's
    (equispaced) times a segment's."""
    return shape is shapes.Wedge


@functools.cache
def wedge_lattice(order):
    """A wedge's nodes of ``order``: ``(i, j, k)``, the triangle's lattice node ``(i, j)``
    (``simplex_lattice``) in layer ``k`` (``0`` to ``order``), node ``t + T * k`` with
    ``T`` the triangle's node count: the triangle fastest."""
    tri = simplex_lattice(shapes.Triangle, order)
    out = np.array([(i, j, k) for k in range(int(order) + 1) for i, j in tri], np.int32)
    out.setflags(write=False)
    return out


def wedge_basis(order, pc):
    """The wedge's Lagrange basis at parametric points ``pc``: the triangle's
    (``simplex_basis``) times the segment's equispaced polynomials in ``z``."""
    pc = np.atleast_2d(np.asarray(pc, np.float64))
    tri = simplex_basis(shapes.Triangle, order, pc[:, :2])
    segment = simplex_basis(_SEGMENT, order, pc[:, 2:3])
    return np.einsum("pt,pk->pkt", tri, segment).reshape(len(pc), -1)


@functools.cache
def wedge_bernstein(order):
    """The matrix taking a wedge's nodal values to its Bernstein coefficients: the
    triangle's and the segment's, one per factor (node ``t + T * k``)."""
    m = np.kron(simplex_bernstein(_SEGMENT, order), simplex_bernstein(shapes.Triangle, order))
    m.setflags(write=False)
    return m


@functools.cache
# ── Pyramids, at levels 1 and 2 ─────────────────────────────────────

def _pyramid_level(level):
    level = int(level)
    if level not in (1, 2):
        raise NotImplementedError(f"a pyramid's lattice is of level 1 or 2 (its corners, "
                                  f"or VTK's order-2 nodes), not {level}")
    return level


def pyramid_lattice(level):
    """A pyramid's lattice nodes as integer collapsed coordinates times ``level``: its
    corners at level 1; at level 2 the first 14 of VTK's order-2 nodes
    (``Pyramid.quadratic_node``: the corners, the edges' middles, the base's centre;
    the triangle faces' centres are no lattice points). An apex edge's middle is at
    its base corner's ``(r, s)``: the coordinates collapse toward the apex."""
    level = _pyramid_level(level)
    rows = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1)]
    if level == 2:
        rows = [(2 * r, 2 * s, 2 * t) for r, s, t in rows] + [
            (1, 0, 0), (2, 1, 0), (1, 2, 0), (0, 1, 0),             # the base's edges
            (0, 0, 1), (2, 0, 1), (2, 2, 1), (0, 2, 1),             # the apex edges
            (1, 1, 0)]                                              # the base's centre
    out = np.array(rows, np.int32)
    out.setflags(write=False)
    return out


def _pyramid_dof_tables(order):
    """``dof_tables`` of a pyramid at level 1 or 2: its corners; then each edge's middle
    (edge ``e``, place 1) and the base's centre (face 0, place (1, 1))."""
    level = _pyramid_level(order)
    n = len(pyramid_lattice(level))
    kind = np.full(n, CORNER, np.int32)
    entity = np.arange(n, dtype=np.int32)
    a = np.zeros(n, np.int32)
    b = np.zeros(n, np.int32)
    if level == 2:
        kind[5:13], entity[5:13], a[5:13] = EDGE, np.arange(8), 1
        kind[13], entity[13], a[13], b[13] = FACE, 0, 1, 1
    tables = (kind, entity, a, b, np.arange(5, dtype=np.int32))
    for t in tables:
        t.setflags(write=False)
    return tables


@functools.cache
def pyramid_nodes(order):
    """An order-``order`` (3 and up) pyramid's nodes, in node order: ``(ijk, points)``, the
    lattice's integer coordinates ``(i, j, k)`` -- layer ``k`` from the base, a grid of
    ``(order - k + 1)^2`` nodes, ``i`` fastest -- and their collapsed coordinates
    ``(i / (order - k), j / (order - k), k / order)`` (the apex ``(0, 0, 1)``), as f64.
    ``(order + 1)(order + 2)(2 order + 3) / 6`` nodes: the corners are nodes ``0``,
    ``order``, ``(order + 1)^2 - 1``, ``order (order + 1)`` and the last."""
    order = int(order)
    if order < 3:
        raise ValueError(f"pyramid_nodes is for orders 3 and up, not {order}")
    ijk, points = [], []
    for k in range(order + 1):
        n = order - k
        for j in range(n + 1):
            for i in range(n + 1):
                ijk.append((i, j, k))
                points.append((i / n if n else 0.0, j / n if n else 0.0, k / order))
    out = (np.array(ijk, np.int32), np.array(points, np.float64))
    for t in out:
        t.setflags(write=False)
    return out


def _pyramid_modes(order, x, y, t):
    """The order-``order`` pyramid space at collapsed coordinates: ``P_i(2x - 1)
    P_j(2y - 1) (1 - t)^m Q_k(2t - 1)``, ``m = max(i, j)``, ``k <= order - m``, with
    Legendre ``P`` and ``Q`` (Bergot, Cohen and Duruflé's space, VTK's ``G`` pyramid's:
    other Jacobi weights span the same), a column a function."""
    from numpy.polynomial import legendre

    def leg(n, u):
        c = np.zeros(n + 1)
        c[n] = 1.0
        return legendre.legval(u, c)

    cols = []
    for i in range(order + 1):
        for j in range(order + 1):
            m = max(i, j)
            for k in range(order - m + 1):
                cols.append(leg(i, 2 * x - 1) * leg(j, 2 * y - 1) * (1 - t) ** m
                            * leg(k, 2 * t - 1))
    return np.stack(cols, axis=1)


@functools.cache
def pyramid_bernstein(order):
    """The order-``order`` pyramid's nodal functions (3 and up; at its ``pyramid_nodes``)
    as tensor Bernstein polynomials in the collapsed coordinates: an ``(N, (order +
    1)^3)`` f64 matrix, row ``k`` node ``k``'s coefficients of ``B_i(r) B_j(s) B_l(t)``
    (degree ``order`` each) at ``i + q j + q^2 l``, ``q = order + 1``. The space's
    functions are of degree at most ``order`` in each collapsed coordinate, so this is
    exact; its values are at the nodes, its bounds the coefficients'."""
    from math import comb

    order = int(order)
    _, nodes = pyramid_nodes(order)
    q = order + 1
    vandermonde = _pyramid_modes(order, *nodes.T)
    nodal = np.linalg.inv(vandermonde)            # modes' weights in each nodal function
    # A tensor grid of interior points the Bernstein polynomials are exact at.
    g = (np.cos(np.pi * (np.arange(q) + 0.5) / q)[::-1] + 1) / 2
    x, y, t = (a.reshape(-1) for a in np.meshgrid(g, g, g, indexing="ij"))
    # Grid point (a, b, c) at a + q b + q^2 c, as the coefficients are numbered.
    x, y, t = (np.ascontiguousarray(a.reshape(q, q, q).transpose(2, 1, 0).reshape(-1))
               for a in (x, y, t))

    def bern(u):
        return np.stack([comb(order, i) * u ** i * (1 - u) ** (order - i) for i in range(q)],
                        axis=1)

    bx, by, bt = bern(x), bern(y), bern(t)
    tensor = (bx[:, None, None, :] * by[:, None, :, None] * bt[:, :, None, None]).reshape(
        len(x), -1)
    values = _pyramid_modes(order, x, y, t) @ nodal
    out = np.linalg.solve(tensor, values).T
    out.setflags(write=False)
    return out


def pyramid_subdivision(level):
    """A pyramid's lattice (``pyramid_lattice``) cut into linear cells: ``(types, rows)``,
    the cells' VTK types and their lattice nodes, flat. Level 1 is the pyramid
    itself; level 2 its uniform refinement, 6 pyramids and 4 tetrahedra (the slab
    under the apex layer: four pyramids on the base's quarters, one upside down
    between them, and the tetrahedra between those; the top pyramid above), so its
    base's quarters stay quads and its triangle faces' quarters triangles, as the
    neighbours' refinements meet them. Each is oriented as VTK's cells are, by its
    volume in the lattice's own (straight) pyramid."""
    level = _pyramid_level(level)
    if level == 1:
        out = (np.array([int(shapes.PYRAMID)], np.int32), np.arange(5, dtype=np.int32))
    else:
        pyramids = [[0, 5, 13, 8, 9], [5, 1, 6, 13, 10], [13, 6, 2, 7, 11], [8, 13, 7, 3, 12],
                    [9, 10, 11, 12, 4], [9, 12, 11, 10, 13]]
        tetrahedra = [[8, 13, 9, 12], [13, 6, 10, 11], [5, 13, 9, 10], [13, 7, 12, 11]]
        # Where each lattice node is in a pyramid of base [0, 2]^2 and apex (1, 1, 2).
        # (the collapsed coordinates r, s scaled toward the apex: (1 - t) r + t / 2).
        ijk = pyramid_lattice(2).astype(float)
        r, s, t = ijk[:, 0] / 2, ijk[:, 1] / 2, ijk[:, 2] / 2
        x = 2 * np.stack([(1 - t) * r + t / 2, (1 - t) * s + t / 2, t], axis=1)
        rows = []
        for p in pyramids:
            base = x[p[:4]]
            normal = np.cross(base[1] - base[0], base[3] - base[0])
            if np.dot(normal, x[p[4]] - base.mean(axis=0)) < 0:
                p = [p[0], p[3], p[2], p[1], p[4]]
            rows.append(p)
        for t in tetrahedra:
            a, b, c, d = x[t]
            if np.dot(np.cross(b - a, c - a), d - a) < 0:
                t = [t[0], t[2], t[1], t[3]]
            rows.append(t)
        out = (np.array([int(shapes.PYRAMID)] * 6 + [int(shapes.TETRA)] * 4, np.int32),
               np.array([n for r in rows for n in r], np.int32))
    for t in out:
        t.setflags(write=False)
    return out


def wedge_subdivision(level):
    """A wedge's order-``level`` lattice as ``level^3`` sub-wedges: each of Freudenthal's
    triangles (``simplex_subdivision``) in each layer, corners as the wedge's own
    (the triangle at the layer's bottom, then at its top)."""
    tri = simplex_subdivision(shapes.Triangle, level)
    count = len(simplex_lattice(shapes.Triangle, level))
    rows = np.array([[*(t + count * k), *(t + count * (k + 1))]
                     for k in range(int(level)) for t in tri], np.int32).reshape(-1, 6)
    rows.setflags(write=False)
    return rows


def _classify(corners, edges, faces, lattice, p):
    """``dof_tables`` by geometry, for a shape whose nodes are ``lattice / p``: each
    node a corner, inside an edge (``a`` steps from its corner 0), inside a face
    (a triangle's ``(a, b)`` toward its corners 1 and 2, a quad's toward 1 and 3),
    or interior (numbered in node order)."""
    corners = np.array(corners, np.int64) * p
    count = len(lattice)
    kind = np.zeros(count, np.int32)
    entity = np.zeros(count, np.int32)
    a = np.zeros(count, np.int32)
    b = np.zeros(count, np.int32)
    corner_node = np.zeros(len(corners), np.int32)
    interior = 0

    def along(x, c0, c1):
        # x = c0 + s (c1 - c0), 0 < s < 1, in steps of 1/p: s * p, or None.
        d = corners[c1] - corners[c0]
        r = x - corners[c0]
        steps = int(r @ d) * p // int(d @ d) if (d @ d) else 0
        return steps if 0 < steps < p and np.array_equal(r * p, d * steps) else None

    def on_face(x, row):
        # In the face's plane and inside it: its steps toward corners 1 and 2
        # (triangle) or 1 and 3 (quad), or None.
        o = corners[row[0]]
        u = corners[row[1]] - o
        v = corners[row[2] if len(row) == 3 else row[3]] - o
        r = x - o
        m = np.c_[u, v].astype(np.float64)
        st, *_ = np.linalg.lstsq(m, r.astype(np.float64), rcond=None)
        if not np.allclose(m @ st, r):
            return None
        sa, sb = (round(float(t) * p) for t in st)
        inside = (sa > 0 and sb > 0 and sa + sb < p) if len(row) == 3 else \
            (0 < sa < p and 0 < sb < p)
        return (sa, sb) if inside else None

    edge_rows = _pairs(edges, 2)
    for n in range(count):
        x = np.asarray(lattice[n], np.int64)
        hit = [c for c in range(len(corners)) if np.array_equal(corners[c], x)]
        if hit:
            kind[n], entity[n] = CORNER, hit[0]
            corner_node[hit[0]] = n
            continue
        for e, (c0, c1) in enumerate(edge_rows):
            steps = along(x, c0, c1)
            if steps is not None:
                kind[n], entity[n], a[n] = EDGE, e, steps
                break
        else:
            for f, row in enumerate(faces):
                place = on_face(x, row)
                if place is not None:
                    kind[n], entity[n] = FACE, f
                    a[n], b[n] = place
                    break
            else:
                kind[n], a[n] = INTERIOR, interior
                interior += 1
    for array in (kind, entity, a, b, corner_node):
        array.setflags(write=False)
    return kind, entity, a, b, corner_node


@functools.cache
def simplex_lattice(shape, order):
    """A simplex's nodes of ``order``: an ``(n, d)`` i32 array of integer coordinates
    ``(i, j, k)``, the node at ``(i, j, k) / order``, with ``i + j + k <= order``,
    in node order: the first axis fastest (``i``, then ``j``, then ``k``). These
    are VTK's and gmsh's equispaced Lagrange nodes, numbered here by place."""
    p = int(order)
    d = len(_SIMPLICES[shape][0][0])
    rows = []
    for k in range(p + 1 if d == 3 else 1):
        for j in range(p + 1 - k if d >= 2 else 1):
            for i in range(p + 1 - j - k):
                rows.append((i, j, k)[:d])
    out = np.array(rows, np.int32).reshape(-1, d)
    out.setflags(write=False)
    return out


def _barycentric(shape, pc):
    """The barycentric coordinates of parametric points ``pc`` (``(n, d)``): corner 0's
    first, ``1 - sum(pc)``."""
    pc = np.atleast_2d(np.asarray(pc, np.float64))
    return np.c_[1.0 - pc.sum(axis=1), pc]


def _multi_indices(shape, order):
    """Each node's barycentric multi-index ``(p - i - j - k, i, j, k)``."""
    lattice = simplex_lattice(shape, order)
    return np.c_[int(order) - lattice.sum(axis=1), lattice]


def simplex_basis(shape, order, pc):
    """The Lagrange basis on the equispaced nodes at parametric points ``pc``: an
    ``(points, nodes)`` array, by Silvester's product formula: node ``alpha``'s
    function is ``prod_m prod_{s < alpha_m} (p lambda_m - s) / (s + 1)``. A host
    reference for the device's evaluation (and the tests')."""
    p = int(order)
    lam = _barycentric(shape, pc)
    alphas = _multi_indices(shape, order)
    out = np.ones((len(lam), len(alphas)))
    for n, alpha in enumerate(alphas):
        for m, a in enumerate(alpha):
            for s in range(a):
                out[:, n] *= (p * lam[:, m] - s) / (s + 1)
    return out


@functools.cache
def simplex_bernstein(shape, order):
    """The matrix taking a polynomial's values at the equispaced nodes to its Bernstein
    coefficients over the simplex (row ``n``, node ``n``'s multi-index): the
    polynomial lies between the least and the greatest (the convex hull
    property), which bounds a field over a cell and a curved cell's geometry."""
    from math import factorial

    p = int(order)
    alphas = _multi_indices(shape, order)
    lam = _barycentric(shape, simplex_lattice(shape, order) / max(p, 1))
    at = np.ones((len(lam), len(alphas)))
    for n, alpha in enumerate(alphas):
        coefficient = factorial(p)
        for a in alpha:
            coefficient //= factorial(a)
        at[:, n] = coefficient * np.prod(lam ** alpha, axis=1)
    m = np.linalg.inv(at)
    m.setflags(write=False)
    return m


@functools.cache
def simplex_subdivision(shape, level):
    """Freudenthal's subdivision of a simplex's order-``level`` lattice: ``level^d``
    sub-simplices, each a row of ``d + 1`` lattice node numbers (``simplex_lattice``'s),
    corner 0 first, wound as the shape's own corners (positive volume). They are
    the Kuhn simplices of a cube lattice in the ordered coordinates ``(i + j + k,
    j + k, k)``: every cube's six (a square's two) paths from its low corner to its
    high one, kept where the path stays in the simplex."""
    import itertools

    p = int(level)
    lattice = simplex_lattice(shape, level)
    d = lattice.shape[1]
    number = {tuple(map(int, row)): n for n, row in enumerate(lattice)}

    def back(u):
        # ordered (u1 >= u2 >= u3) to lattice (i, j, k)
        u = [*u, 0]
        return tuple(u[m] - u[m + 1] for m in range(d))

    rows = []
    for low in itertools.product(range(p), repeat=d):
        for axes in itertools.permutations(range(d)):
            path = [list(low)]
            for axis in axes:
                step = list(path[-1])
                step[axis] += 1
                path.append(step)
            if all(all(u[m] >= u[m + 1] for m in range(d - 1)) and u[0] <= p for u in path):
                rows.append([number[back(u)] for u in path])
    rows = np.array(rows, np.int32).reshape(-1, d + 1)
    # Wind each as the shape's corners: positive (signed) measure.
    corners = np.array(_SIMPLICES[shape][0], np.float64)
    for r in rows:
        x = lattice[r].astype(np.float64)
        if np.linalg.det(x[1:] - x[0]) < 0:
            r[1], r[2] = r[2], r[1]
    assert len(rows) == p ** d, (shape, level, len(rows))
    assert np.linalg.det(corners[1:] - corners[0]) > 0
    rows.setflags(write=False)
    return rows


@functools.cache
def lattice_ijk(shape, level):
    """Each node of a ``shape`` cell's order-``level`` lattice as integer coordinates,
    in node order: ``(n, d)`` i32, the tensor lattice's ``(i, j, k)`` with the first
    axis fastest, or the simplex's (``simplex_lattice``), or a pyramid's at level 1
    or 2 (``pyramid_lattice``)."""
    if shape in _SIMPLICES:
        return simplex_lattice(shape, level)
    if shape is shapes.Wedge:
        return wedge_lattice(level)
    if shape is shapes.Pyramid:
        return pyramid_lattice(level)
    corners = _SHAPES[shape][0]
    d = len(corners[0])
    q = int(level) + 1
    out = np.array([[(n // q ** axis) % q for axis in range(d)] for n in range(q ** d)],
                   np.int32).reshape(-1, d)
    out.setflags(write=False)
    return out


def _simplex_entity_nodes(shape, order):
    """``entity_nodes`` for a simplex: each edge's ``order + 1`` nodes from its corner 0,
    and each triangle face's ``(order + 1)(order + 2) / 2`` in its own lattice order
    (``a`` steps toward its corner 1 fastest, ``b`` toward its corner 2)."""
    corners, edge_table, faces = _SIMPLICES[shape]
    p = int(order)
    alphas = _multi_indices(shape, order)
    number = {tuple(map(int, alpha)): n for n, alpha in enumerate(alphas)}
    d = len(corners[0])

    def node(weights):
        alpha = [0] * (d + 1)
        for corner, w in weights.items():
            alpha[corner] += w
        return number[tuple(alpha)]

    edges = np.array([[node({c0: p - j, c1: j}) for j in range(p + 1)]
                      for c0, c1 in _pairs(edge_table, 2)], np.int32).reshape(-1, p + 1)
    tri = simplex_lattice(shapes.Triangle, order)
    face_rows = [[node({row[0]: p - a - b, row[1]: a, row[2]: b}) for a, b in tri]
                 for row in faces]
    faces_out = np.array(face_rows, np.int32).reshape(len(faces), -1) if faces else \
        np.zeros((0, len(tri)), np.int32)
    return edges, faces_out


def _wedge_entity_nodes(order):
    """``entity_nodes`` for a wedge: each edge's ``order + 1`` nodes from its corner 0;
    each face's nodes, a row of ``(order + 1)^2`` (a triangle's
    ``(order + 1)(order + 2) / 2`` first, in its lattice order from its corner 0
    toward 1 and 2; a quad's from its corner 0 toward 1 fastest, then 3)."""
    corners, edge_table, faces = _WEDGE
    p = int(order)
    lattice = wedge_lattice(order)
    number = {tuple(map(int, row)): n for n, row in enumerate(lattice)}
    c = np.array(corners, np.int64) * p

    def node(x):
        return number[tuple(int(v) for v in x)]

    edges = np.array([[node(c[c0] + (c[c1] - c[c0]) * j // p) for j in range(p + 1)]
                      for c0, c1 in _pairs(edge_table, 2)], np.int32)
    rows = np.zeros((len(faces), (p + 1) ** 2), np.int32)
    tri = simplex_lattice(shapes.Triangle, order)
    for f, row in enumerate(faces):
        if len(row) == 3:
            nodes = [node(c[row[0]] + ((c[row[1]] - c[row[0]]) * a + (c[row[2]] - c[row[0]]) * b)
                          // p) for a, b in tri]
        else:
            nodes = [node(c[row[0]] + ((c[row[1]] - c[row[0]]) * a + (c[row[3]] - c[row[0]]) * b)
                          // p) for b in range(p + 1) for a in range(p + 1)]
        rows[f, :len(nodes)] = nodes
    return edges, rows


def _simplex_dof_tables(shape, order):
    """``dof_tables`` for a simplex (see there): a face's ``(a, b)`` from its corner 0
    toward its corners 1 and 2; interior nodes numbered in node order."""
    corners, edge_table, faces = _SIMPLICES[shape]
    p = int(order)
    d = len(corners[0])
    edges = _pairs(edge_table, 2)
    lattice = simplex_lattice(shape, order)
    alphas = _multi_indices(shape, order)
    count = len(lattice)
    kind = np.zeros(count, np.int32)
    entity = np.zeros(count, np.int32)
    a = np.zeros(count, np.int32)
    b = np.zeros(count, np.int32)
    corner_node = np.zeros(len(corners), np.int32)
    interior = 0
    for n in range(count):
        alpha = alphas[n]                       # alpha[m]: weight of corner m, times p
        held = [m for m in range(d + 1) if alpha[m] > 0]
        if len(held) == 1:
            kind[n], entity[n] = CORNER, held[0]
            corner_node[held[0]] = n
        elif len(held) == 2:
            for e, (c0, c1) in enumerate(edges):
                if {c0, c1} == set(held):
                    kind[n], entity[n], a[n] = EDGE, e, alpha[c1]
                    break
        elif len(held) == 3 and d == 3:
            for f, row in enumerate(faces):
                if set(row) == set(held):
                    kind[n], entity[n] = FACE, f
                    a[n], b[n] = alpha[row[1]], alpha[row[2]]
                    break
        else:
            kind[n], a[n] = INTERIOR, interior
            interior += 1
    for array in (kind, entity, a, b, corner_node):
        array.setflags(write=False)
    return kind, entity, a, b, corner_node


def _pairs(table, k):
    flat = [int(v) for v in np.asarray(table).reshape(-1)]
    return [tuple(flat[i:i + k]) for i in range(0, len(flat), k)]


@functools.cache
def _entities(shape):
    """A shape's corners (integer parametric coordinates), edges (corner pairs) and
    faces (rows of corners, triangles and quads), as ``dof_tables`` reads them."""
    if shape in _SIMPLICES:
        corners, edge_table, faces = _SIMPLICES[shape]
        return corners, _pairs(edge_table, 2), [tuple(f) for f in faces]
    if shape is shapes.Wedge:
        corners, edge_table, faces = _WEDGE
        return corners, _pairs(edge_table, 2), [tuple(f) for f in faces]
    corners, edge_table, face_table = _SHAPES[shape]
    return (corners, _pairs(edge_table, 2) if len(edge_table) else [],
            _pairs(face_table, 4) if len(face_table) else [])


@functools.cache
def entity_coordinates(shape, level):
    """Where each node of a ``shape`` cell's order-``level`` lattice (``lattice_ijk``)
    lies on the cell's closed edges and faces: ``(edges, faces)``, i32 arrays of
    ``(n, E)`` steps from each edge's corner 0 (-1 off it) and ``(n, F, 2)`` steps
    from each face's corner 0 toward its corners 1 and 2 (a triangle) or 1 and 3
    (a quad) (-1 off it). Two nodes on one edge are joined along it, and two on
    one face across it, which is how a sub-edge finds its entity."""
    corners, edges, faces = _entities(shape)
    p = int(level)
    lattice = np.asarray(lattice_ijk(shape, level), np.int64)
    corners = np.array(corners, np.int64) * p
    count = len(lattice)
    on_edges = np.full((count, max(len(edges), 1)), -1, np.int32)
    on_faces = np.full((count, max(len(faces), 1), 2), -1, np.int32)
    for n in range(count):
        x = lattice[n]
        for e, (c0, c1) in enumerate(edges):
            d = corners[c1] - corners[c0]
            r = x - corners[c0]
            steps = int(r @ d) * p // int(d @ d)
            if 0 <= steps <= p and np.array_equal(r * p, d * steps):
                on_edges[n, e] = steps
        for f, row in enumerate(faces):
            o = corners[row[0]]
            u = corners[row[1]] - o
            v = corners[row[2] if len(row) == 3 else row[3]] - o
            m = np.c_[u, v].astype(np.float64)
            st, *_ = np.linalg.lstsq(m, (x - o).astype(np.float64), rcond=None)
            if not np.allclose(m @ st, x - o):
                continue
            sa, sb = (round(float(t) * p) for t in st)
            inside = (sa >= 0 and sb >= 0 and sa + sb <= p) if len(row) == 3 else \
                (0 <= sa <= p and 0 <= sb <= p)
            if inside:
                on_faces[n, f] = sa, sb
    for array in (on_edges, on_faces):
        array.setflags(write=False)
    return on_edges, on_faces


def dof_tables(shape, order):
    """For each of a ``shape`` cell's ``(order + 1)^d`` nodes, in node order, where its
    value lives: four i32 arrays ``(kind, entity, a, b)``.

    - ``CORNER``: ``entity`` is the cell's corner (its point).
    - ``EDGE``: ``entity`` is the cell's local edge and ``a`` (1 to ``order - 1``)
      the node's place along it from the edge's corner 0.
    - ``FACE``: ``entity`` is the cell's local face, ``(a, b)`` the node's place
      from the face's corner 0, ``a`` toward its corner 1 and ``b`` toward its
      corner 3.
    - ``INTERIOR``: ``a`` is the node's place among the cell's interior nodes,
      lexicographic, the first axis fastest.

    Also ``corner_node``: corner ``j``'s node, an i32 array of ``NUM_POINTS``.

    A simplex's nodes are ``simplex_lattice``'s, and its faces triangles: a face
    node's ``(a, b)`` counts steps from the face's corner 0 toward its corners 1
    and 2 (see ``_simplex_dof_tables``)."""
    if shape in _SIMPLICES:
        return _simplex_dof_tables(shape, order)
    if shape is shapes.Wedge:
        corners, edges, faces = _WEDGE
        return _classify(corners, edges, faces, wedge_lattice(order), int(order))
    if shape is shapes.Pyramid:
        return _pyramid_dof_tables(order)
    if shape not in _SHAPES:
        raise NotImplementedError(f"bases of any order are on segments, quads, pixels, "
                                  f"hexahedra and voxels, not {shape.__name__}")
    corners, edge_table, face_table = _SHAPES[shape]
    p = int(order)
    d = len(corners[0])
    edges = _pairs(edge_table, 2) if len(edge_table) else []
    faces = _pairs(face_table, 4) if len(face_table) else []
    q = p + 1
    count = q ** d
    kind = np.zeros(count, np.int32)
    entity = np.zeros(count, np.int32)
    a = np.zeros(count, np.int32)
    b = np.zeros(count, np.int32)
    corner_node = np.zeros(len(corners), np.int32)

    def place(index, start, toward):
        """The node's steps from corner ``start`` toward corner ``toward``."""
        axis = next(i for i in range(d) if corners[start][i] != corners[toward][i])
        return index[axis] if corners[start][axis] == 0 else p - index[axis]

    for n in range(count):
        index = [(n // q ** axis) % q for axis in range(d)]
        ends = [i in (0, p) for i in index]
        if all(ends):
            at = tuple(i // p for i in index) if p else tuple(0 for _ in index)
            kind[n], entity[n] = CORNER, corners.index(at)
            corner_node[corners.index(at)] = n
        elif not any(ends):
            interior = [i - 1 for i in index]
            kind[n] = INTERIOR
            a[n] = sum(interior[axis] * (p - 1) ** axis for axis in range(d))
        elif d - sum(ends) == 1:
            # On an edge: the corners it runs between agree with the node's ends.
            free = ends.index(False)
            for e, (c0, c1) in enumerate(edges):
                if all(corners[c0][i] * p == index[i] for i in range(d) if i != free) \
                        and corners[c0][free] != corners[c1][free]:
                    kind[n], entity[n], a[n] = EDGE, e, place(index, c0, c1)
                    break
            else:
                raise AssertionError(f"node {n} of {shape.__name__} is on no edge")
        else:
            # On a face of a solid: the face whose corners share the node's end.
            fixed = ends.index(True)
            for f, row in enumerate(faces):
                if all(corners[c][fixed] * p == index[fixed] for c in row):
                    kind[n], entity[n] = FACE, f
                    a[n], b[n] = place(index, row[0], row[1]), place(index, row[0], row[3])
                    break
            else:
                raise AssertionError(f"node {n} of {shape.__name__} is on no face")
    for array in (kind, entity, a, b, corner_node):
        array.setflags(write=False)
    return kind, entity, a, b, corner_node


def view_order(shape, order, any_order=False):
    """Node ``n`` (lexicographic) as an ``H1`` view of ``order`` numbers its cell's nodes
    (``dof_index``): itself above order 2, and with ``any_order`` (a view reading
    the space as a basis of any order at 2 as well: a Bernstein basis, an
    element-to-DOF table; ``H1._any_order``); at order 2 the quadratic shapes'
    order -- corners, edges, quad faces, interior (``shapes.Shape.quadratic_node``)."""
    kind, entity, _, _, _ = dof_tables(shape, order)
    if order != 2 or any_order:
        return np.arange(len(kind), dtype=np.int32)
    first_face = shape.NUM_FACES - shape.QUADRATIC_FACES
    return np.array([e if k == CORNER else
                     shape.NUM_POINTS + e if k == EDGE else
                     shape.NUM_POINTS + shape.NUM_EDGES + e - first_face if k == FACE else
                     shape.NUM_POINTS + shape.NUM_EDGES + shape.QUADRATIC_FACES
                     for k, e in zip(kind, entity)], np.int32)


@functools.cache
def entity_nodes(shape, order):
    """A ``shape`` cell's nodes of ``order`` on each of its edges and quad faces, as
    lexicographic node numbers: ``(edges, faces)``, ``edges[e, j]`` the ``j``-th node
    from local edge ``e``'s corner 0 to its corner 1, ``faces[f, a + (order + 1) *
    b]`` node ``(a, b)`` of local face ``f`` from its corner 0, ``a`` toward its corner 1
    and ``b`` toward its corner 3. A field's values on an edge or face are its trace
    there, which these name."""
    if shape in _SIMPLICES:
        return _simplex_entity_nodes(shape, order)
    if shape is shapes.Wedge:
        return _wedge_entity_nodes(order)
    if shape not in _SHAPES:
        raise NotImplementedError(f"no tensor nodes on {shape.__name__}")
    corners, edge_table, face_table = _SHAPES[shape]
    p = int(order)
    q = p + 1
    d = len(corners[0])

    def number(coordinates):
        return sum(int(c) * q ** axis for axis, c in enumerate(coordinates))

    def at(c0, toward, step):
        return [corners[c0][i] * p + (corners[toward][i] - corners[c0][i]) * step
                for i in range(d)]

    edges = np.array([[number(at(c0, c1, j)) for j in range(q)]
                      for c0, c1 in (_pairs(edge_table, 2) if len(edge_table) else [])],
                     np.int32).reshape(-1, q)
    faces = []
    for row in (_pairs(face_table, 4) if len(face_table) else []):
        nodes = []
        for b in range(q):
            for a in range(q):
                c = [corners[row[0]][i] * p + (corners[row[1]][i] - corners[row[0]][i]) * a
                     + (corners[row[3]][i] - corners[row[0]][i]) * b for i in range(d)]
                nodes.append(number(c))
        faces.append(nodes)
    faces = np.array(faces, np.int32).reshape(-1, q * q)
    edges.setflags(write=False)
    faces.setflags(write=False)
    return edges, faces


def view_index(shape, order, any_order=False):
    """Lexicographic node ``n``'s number in an ``H1`` view of ``order``'s ``dof(c, k)``:
    at order 1 the corner it is (point data reads corners), else ``view_order``."""
    if order == 1:
        return np.asarray(dof_tables(shape, 1)[1], np.int32)
    return view_order(shape, order, any_order)
