# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""VTK cell grids (``vtkCellGrid``, the discontinuous-Galerkin cells of VTK's
Filters/CellGrid) as Tack datasets.

A cell grid keeps, per cell type (``vtkDGHex``, ...), a corner connectivity and,
per attribute and cell type, its function space, basis, order and DOF sharing
(``vtkCellAttribute``'s cell-type information, readable from Python since VTK's
``GetCellTypes``/``GetDOFSharing``/... accessors). Tack reads it so:

- Cells: each type's cells, types in VTK's cell order (by their cell specs'
  offsets), on the cell spec's corner ids (VTK's DG corner orders are its classic
  cells', which are Tack's).
- The shape attribute (geometry) must be ``HGRAD`` and shared. Its order-1
  complete basis (``C1``) is the points themselves, copied; a higher order is
  resampled onto ``H1(order=p, basis="equispaced")``.
- ``constant`` attributes are ``Constant`` fields.
- ``HGRAD`` ``C1``: shared on the shape's points, an ``H1`` field of the group's
  values; per cell, an ``L2`` field of its corner values, copied.
- Every other ``HGRAD`` basis (``C2``, serendipity ``I2``, the Gauss-point ``G``
  bases, arbitrary-order ``A``) and ``Bezier``: VTK evaluates the attribute at
  the nodes of Tack's equispaced space of that order (``vtkCellGridEvaluator``'s
  interpolate-only mode), per cell into ``L2``, shared into ``H1``. Where the two
  spaces are the same polynomials (every one but the enriched ``F`` bases and
  pyramids' other bases, which are left out) this is exact, to rounding.
- Pyramids' ``HGRAD`` ``C2`` (VTK's 18-node pyramid, Tack's order 2 there), the
  shape's or an attribute's, shared or per cell: copied node by node, each VTK
  node matched to Tack's by its reference coordinates (VTK's ``C2`` basis numbers
  its corners from ``(1, 1, 0)``, not as the cell's). Copying takes a coefficient
  as its node's value, as the miniapps write them; VTK's own ``C2`` pyramid basis
  is 8/9 at its triangle faces' nodes (a VTK bug), so VTK draws such data a little
  differently inside a pyramid.
- ``HCURL`` and ``HDIV`` have no Tack space: left out, with a warning.

Reference coordinates: VTK's tensor axes (an edge's, a quad's, a hexahedron's,
a wedge's ``t``) are ``[-1, 1]``, Tack's ``[0, 1]``; simplices share the unit
simplex; Tack's pyramid is the unit cube with its top collapsed, VTK's a base on
``[-1, 1]^2`` under an apex at ``(0, 0, 1)``.
"""

import warnings

import numpy as np

import tack
from tack.dataset import shapes

#: VTK's DG cell types and Tack's shapes.
SHAPES = {"vtkDGVert": shapes.Vertex, "vtkDGEdge": shapes.Line, "vtkDGTri": shapes.Triangle,
          "vtkDGQuad": shapes.Quad, "vtkDGTet": shapes.Tetra, "vtkDGHex": shapes.Hexahedron,
          "vtkDGWdg": shapes.Wedge, "vtkDGPyr": shapes.Pyramid}

_registered = False


def _register():
    """VTK's DG cells and their responders, once (queries return nothing without)."""
    global _registered
    if not _registered:
        from vtkmodules.vtkFiltersCellGrid import vtkFiltersCellGrid

        vtkFiltersCellGrid.RegisterCellsAndResponders()
        _registered = True


def _text(token):
    """A ``vtkStringToken``'s string ("" for the empty token)."""
    return token if isinstance(token, str) else (token.Data() if token.HasData() else "")


class _Info:
    """An attribute's cell-type information for one cell type, with its arrays."""

    def __init__(self, attribute, cell_type):
        self.sharing = _text(attribute.GetDOFSharing(cell_type))
        self.space = _text(attribute.GetFunctionSpace(cell_type))
        self.basis = _text(attribute.GetBasis(cell_type))
        self.order = int(attribute.GetOrder(cell_type))
        values = attribute.GetArrayForCellTypeAndRole(cell_type, "values")
        connectivity = attribute.GetArrayForCellTypeAndRole(cell_type, "connectivity")
        self.values = None if values is None else np.asarray(values)
        self.values_name = None if values is None else values.GetName()
        self.connectivity = None if connectivity is None else \
            np.asarray(connectivity).reshape(connectivity.GetNumberOfTuples(), -1)
        self.components = attribute.GetNumberOfComponents()

    def key(self):
        return self.sharing != "", self.space, self.basis, self.order


def _to_vtk_parameters(shape, points):
    """Tack's parametric points of ``shape`` (``(n, d)``) as VTK's reference
    coordinates (``(n, 3)``): tensor axes from ``[0, 1]`` to ``[-1, 1]``."""
    points = np.asarray(points, np.float64)
    out = np.zeros((len(points), 3))
    out[:, :points.shape[1]] = points
    if shape in (shapes.Line, shapes.Quad, shapes.Hexahedron):
        out[:, :shape.DIMENSION] = 2.0 * out[:, :shape.DIMENSION] - 1.0
    elif shape is shapes.Wedge:
        out[:, 2] = 2.0 * out[:, 2] - 1.0
    elif shape is shapes.Pyramid:
        # Tack's pyramid is the unit cube with its top collapsed; VTK's has its base
        # on [-1, 1]^2 at t = 0 and its apex at (0, 0, 1).
        z = out[:, 2].copy()
        out[:, 0] = (2.0 * out[:, 0] - 1.0) * (1.0 - z)
        out[:, 1] = (2.0 * out[:, 1] - 1.0) * (1.0 - z)
    return out


def _lattice(shape, order):
    """Tack's equispaced nodes of ``shape`` at ``order``, in node order, as its
    parametric points; at order 1 its corners, in corner order (the order of an
    order-1 field's values)."""
    from tack.dataset import bases

    lattice = np.asarray(bases.lattice_ijk(shape, order), np.float64) / order
    if order == 1:
        kind, entity, _, _, _ = bases.dof_tables(shape, 1)
        corner_node = np.empty(shape.NUM_POINTS, np.int64)
        corner_node[entity[kind == bases.CORNER]] = np.flatnonzero(kind == bases.CORNER)
        return lattice[corner_node]
    return lattice


def evaluate(grid, attribute, cell_type, cells, parameters):
    """``attribute`` evaluated by VTK in cells ``cells`` (indices among
    ``cell_type``'s) at VTK reference coordinates ``parameters`` (``(n, 3)``):
    ``(n, components)`` doubles."""
    from vtkmodules.vtkCommonCore import (
        vtkDoubleArray,
        vtkStringToken,
        vtkTypeUInt32Array,
        vtkTypeUInt64Array,
    )
    from vtkmodules.vtkCommonDataModel import vtkCellGridEvaluator

    _register()
    types = vtkTypeUInt32Array()
    types.InsertNextValue(vtkStringToken(cell_type).GetHash())
    offsets = vtkTypeUInt64Array()
    offsets.InsertNextValue(0)
    indices = vtkTypeUInt64Array()
    indices.SetNumberOfTuples(len(cells))
    np.asarray(indices)[:] = np.asarray(cells, np.uint64)
    points = vtkDoubleArray()
    points.SetNumberOfComponents(3)
    points.SetNumberOfTuples(len(cells))
    np.asarray(points)[:] = parameters
    query = vtkCellGridEvaluator()
    query.SetCellAttribute(attribute)
    query.InterpolateCellParameters(types, offsets, indices, points)
    grid.Query(query)
    values = np.asarray(query.GetInterpolatedValues())
    if len(values) != len(cells):
        raise RuntimeError("VTK's cell grid evaluator interpolated no values: interpolating "
                           "given cells and parameters needs a VTK with its fix "
                           "(vtkCellGridEvaluator's interpolate-only mode)")
    return values.reshape(len(cells), -1)


def _resampled(grid, attribute, cell_types, counts, order):
    """``attribute`` at Tack's equispaced nodes of ``order`` in every cell, cell by
    cell in ``cell_types`` order: ``(cells' nodes, components)``, each cell's in its
    node order, and the nodes a cell of each type has."""
    parts, per_cell = [], []
    for cell_type, count in zip(cell_types, counts, strict=True):
        shape = SHAPES[cell_type]
        lattice = _lattice(shape, order)
        per_cell.append(len(lattice))
        if not count:
            continue
        cells = np.repeat(np.arange(count), len(lattice))
        parameters = np.tile(_to_vtk_parameters(shape, lattice), (count, 1))
        parts.append(evaluate(grid, attribute, cell_type, cells, parameters))
    width = attribute.GetNumberOfComponents()
    return (np.concatenate(parts) if parts else np.zeros((0, width))), per_cell


#: The reference coordinates of VTK's C2 pyramid nodes (``Basis/HGrad/PyrC2Basis.h``),
#: in its basis' order: the corners from ``(1, 1, 0)`` round the base, the apex, the
#: base's edges' and the apex edges' middles in that corner order, the base's
#: centre, the triangle faces' centroids. Found by evaluating the basis: each
#: function is 1 at its node (but the face functions, 8/9: a VTK bug).
_PYRAMID_C2 = np.array([
    (1, 1, 0), (-1, 1, 0), (-1, -1, 0), (1, -1, 0), (0, 0, 1),
    (0, 1, 0), (-1, 0, 0), (0, -1, 0), (1, 0, 0),
    (0.5, 0.5, 0.5), (-0.5, 0.5, 0.5), (-0.5, -0.5, 0.5), (0.5, -0.5, 0.5), (0, 0, 0),
    (0, 2 / 3, 1 / 3), (-2 / 3, 0, 1 / 3), (0, -2 / 3, 1 / 3), (2 / 3, 0, 1 / 3)])
#: vtkDGPyr's corners (its C1 basis' nodes).
_PYRAMID_C1 = np.array([(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0), (0, 0, 1)], float)
#: Tack's order-2 pyramid nodes by corners: the corners, the edges (shapes' order),
#: the base's centre, the triangle faces (shapes.Pyramid.quadratic_node).
_TACK_PYRAMID = ([(j,) for j in range(5)]
                 + [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 4), (2, 4), (3, 4)]
                 + [(0, 1, 2, 3), (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)])


def _pyramid_corner_parameters(shape_info, corners):
    """Where Tack's corners of a grid's pyramids are in VTK's reference pyramid:
    ``(5, 3)``, each found by its point among the shape's own nodes (whose places
    its basis gives: C1's are vtkDGPyr's corners, C2's ``_PYRAMID_C2``). The same
    for every cell, or refused."""
    connectivity = shape_info.connectivity
    places = _PYRAMID_C2 if shape_info.order == 2 else _PYRAMID_C1
    columns = np.empty((len(corners), 5), np.int64)
    for j in range(5):
        hits = connectivity[:, :5] == corners[:, j:j + 1]
        if not hits.any(axis=1).all():
            raise NotImplementedError("a pyramid whose corners are not its shape's corners")
        columns[:, j] = hits.argmax(axis=1)
    if len(columns) and not (columns == columns[0]).all():
        raise NotImplementedError("pyramids whose corners sit differently in their shape "
                                  "from cell to cell")
    return places[columns[0] if len(columns) else np.arange(5)]


def _pyramid_corner_order(shape_info, corners):
    """The columns of ``corners`` (a grid's pyramids' corners, ``(n, 5)``) that put its
    corner ``j`` at vtkDGPyr's corner ``j``."""
    places = _pyramid_corner_parameters(shape_info, corners)
    order = [int(np.flatnonzero(np.abs(places - target).max(axis=1) < 1e-9)[0])
             for target in _PYRAMID_C1]
    if sorted(order) != list(range(5)):
        raise NotImplementedError("a pyramid's corners are not its shape's corners")
    return order


def _pyramid_node_order(corner_parameters):
    """For each of Tack's 18 order-2 pyramid nodes, the VTK C2 node at its reference
    coordinates (Tack's nodes placed by its corners' VTK coordinates: an edge's
    middle, the base's centre, a face's centroid)."""
    order = np.empty(18, np.int64)
    for k, by in enumerate(_TACK_PYRAMID):
        place = corner_parameters[list(by)].mean(axis=0)
        found = np.flatnonzero(np.abs(_PYRAMID_C2 - place).max(axis=1) < 1e-9)
        if len(found) != 1:
            raise NotImplementedError("a pyramid's corners do not make VTK's reference "
                                      "pyramid")
        order[k] = found[0]
    return order


def _copied_pyramids(grid, attribute, info, cell_types, counts, corners, shape_infos=None):
    """An attribute at Tack's order-2 nodes, cell by cell in ``cell_types`` order, as
    ``_resampled`` gives it: its pyramids' C2 coefficients copied node by node
    (``_pyramid_node_order``), its other cell types resampled by VTK at order 2;
    ``info`` its per-type ``_Info`` (the shape's when ``shape_infos`` is ``None``)."""
    shape_infos = shape_infos or info
    parts, per_cell = [], []
    for k, (cell_type, count) in enumerate(zip(cell_types, counts, strict=True)):
        shape = SHAPES[cell_type]
        if shape is not shapes.Pyramid:
            values, sizes = _resampled(grid, attribute, [cell_type], [count], 2)
            parts.append(values)
            per_cell += sizes
            continue
        per_cell.append(18)
        if not count:
            continue
        order = _pyramid_node_order(_pyramid_corner_parameters(shape_infos[cell_type],
                                                               corners[k]))
        width = attribute.GetNumberOfComponents()
        mine = info[cell_type]
        if mine.sharing:
            values = mine.values.reshape(-1, width)[mine.connectivity[:, order]]
        else:
            values = mine.values.reshape(count, 18, width)[:, order]
        parts.append(values.reshape(-1, width))
    width = attribute.GetNumberOfComponents()
    return (np.concatenate(parts) if parts else np.zeros((0, width))), per_cell


def _corners(grid, cell_type, shape_info):
    """``cell_type``'s cells' corner ids: its cell spec's first corners, or (without
    one) those of a corner-first shape basis's connectivity."""
    spec = grid.GetCellType(cell_type).GetCellSourceConnectivity(-1)
    if spec is not None:
        # Corner-first (VTK's sides read a cell spec's first corners): a spec may
        # share a quadratic shape's connectivity.
        rows = np.asarray(spec).reshape(spec.GetNumberOfTuples(), -1)
        return rows[:, :SHAPES[cell_type].NUM_POINTS]
    if shape_info.basis not in ("C", "I"):
        raise NotImplementedError(f"{cell_type} cells without a cell spec, their shape in basis "
                                  f"{shape_info.basis}: no corners to read")
    return shape_info.connectivity[:, :SHAPES[cell_type].NUM_POINTS]


def to_dataset(grid, dtype=tack.f32, id_dtype=None, shift=(0.0, 0.0, 0.0)):
    """A ``vtkCellGrid`` as a ``DataSet`` (see the module's docstring)."""
    from tack.dataset import H1, L2, Constant, DataSet, Field, UnstructuredTopology, ids
    from tack.interop.mfem import _h1_indices

    _register()
    # VTK numbers cells type by type from each cell spec's offset (its first id):
    # Tack's cells follow that order.
    cell_types = sorted((t for t in map(_text, grid.GetCellTypes()) if t in SHAPES),
                        key=lambda t: grid.GetCellType(t).GetCellSourceOffset(-1))
    unknown = [t for t in map(_text, grid.GetCellTypes()) if t not in SHAPES]
    if unknown:
        raise NotImplementedError(f"cell grids of {', '.join(unknown)} cells")
    counts = [grid.GetCellType(t).GetNumberOfCells() for t in cell_types]
    shape_attribute = grid.GetShapeAttribute()
    if shape_attribute is None:
        raise ValueError("a cell grid without a shape attribute has no geometry")
    infos = {t: _Info(shape_attribute, t) for t in cell_types}
    for t, info in infos.items():
        if info.space != "HGRAD" or not info.sharing:
            raise NotImplementedError(
                f"a cell grid's shape is an HGRAD attribute with shared values, not "
                f"{info.space or 'none'} {'shared' if info.sharing else 'per cell'} on {t}")
        if info.order > 1 and SHAPES[t] is shapes.Pyramid and (info.basis, info.order) != ("C", 2):
            raise NotImplementedError(
                f"a curved pyramid ({info.basis}{info.order}): Tack's curved pyramids are "
                "VTK's C2 (18 nodes)")
        if info.connectivity is None:
            raise NotImplementedError(f"a cell grid's shape on {t} without a connectivity")
    if len({infos[t].values_name for t in cell_types}) > 1:
        raise NotImplementedError("a cell grid whose cell types' shapes keep their points "
                                  "in different arrays")
    points = infos[cell_types[0]].values.reshape(-1, 3) if cell_types else np.zeros((0, 3))
    corners = [_corners(grid, t, infos[t]) for t in cell_types]
    for k, t in enumerate(cell_types):
        if SHAPES[t] is shapes.Pyramid and counts[k]:
            # Tack's corner j at vtkDGPyr's corner j, whichever order the cell spec
            # lists them in (a C2 shape's own lists them as its basis does, from
            # (1, 1, 0)): then Tack's reference pyramid is VTK's, for every attribute.
            corners[k] = corners[k][:, _pyramid_corner_order(infos[t], corners[k])]
    sizes = [SHAPES[t].NUM_POINTS for t in cell_types]
    n = sum(counts)
    connectivity = np.concatenate([c.reshape(-1) for c in corners]) if corners else \
        np.zeros(0, np.int64)
    order = max(info.order for info in infos.values()) if infos else 1
    used = None
    if order > 1:
        # A curved shape's other nodes are not points: the topology keeps the corners
        # (renumbered), and the geometry is resampled at Tack's nodes below.
        used, connectivity = np.unique(connectivity, return_inverse=True)
        points = points[used]
    idt = ids.at_least(id_dtype or tack.i32, max(len(points), len(connectivity)))
    offsets = np.concatenate([[0], np.cumsum(np.repeat(sizes, counts))]) if n else \
        np.zeros(1, np.int64)
    types = np.repeat([int(SHAPES[t].ID) for t in cell_types], counts).astype(np.uint8)
    topology = UnstructuredTopology(tack.array(types), tack.array(offsets.astype(
        idt.numpy_dtype)), tack.array(connectivity.astype(idt.numpy_dtype)),
        num_points=len(points), id_dtype=idt)
    vec = tack.vec(3, dtype)
    positions = tack.empty((len(points),), vec)
    tack.copyto(positions, (points - np.asarray(shift)).astype(dtype.numpy_dtype))
    data = DataSet(topology, positions, origin=shift)
    if order > 1:
        space = H1(data, order=order, basis="equispaced")
        if any(SHAPES[t] is shapes.Pyramid for t in cell_types):
            nodes, per_cell = _copied_pyramids(grid, shape_attribute, infos, cell_types,
                                               counts, corners)
        else:
            nodes, per_cell = _resampled(grid, shape_attribute, cell_types, counts, order)
        if len(set(per_cell)) > 1:
            raise NotImplementedError("a curved cell grid of several cell types")
        values = np.zeros((space.size, 3))
        values[_h1_indices(data, space, per_cell[0]).reshape(-1)] = nodes - np.asarray(shift)
        geometry = tack.empty((space.size,), vec)
        tack.copyto(geometry, values.astype(dtype.numpy_dtype))
        data.geometry = Field(space, geometry)

    left_out = []
    for attribute in grid.GetCellAttributeList():
        if attribute is shape_attribute or _text(attribute.GetName()) == \
                _text(shape_attribute.GetName()):
            continue
        name = _text(attribute.GetName())
        present = [t for t in cell_types if t in map(_text, attribute.GetCellTypes())]
        if present != cell_types:
            left_out.append(f"{name} (not on every cell type)")
            continue
        info = {t: _Info(attribute, t) for t in cell_types}
        keys = {i.key() for i in info.values()}
        if len(keys) > 1:
            left_out.append(f"{name} (its cell types in different spaces)")
            continue
        shared, space_name, basis, field_order = keys.pop()
        width = attribute.GetNumberOfComponents()
        element = dtype if width == 1 else tack.vec(width, dtype)
        if space_name == "constant":
            values = np.concatenate([i.values.reshape(-1, width) for i in info.values()])
            data.fields[name] = Field(Constant(data), _array(values, element, dtype))
        elif space_name == "HGRAD" and basis == "C" and field_order == 1:
            if shared:
                same = all(np.array_equal(info[t].connectivity[:, :SHAPES[t].NUM_POINTS],
                                          corners[k]) for k, t in enumerate(cell_types))
                values = info[cell_types[0]].values.reshape(-1, width)
                values = values if used is None or len(values) <= used.max() else values[used]
                if not same or len(values) != len(points):
                    left_out.append(f"{name} (shared on other points than the shape's)")
                    continue
                data.fields[name] = Field(H1(data), _array(values, element, dtype))
            else:
                values = np.concatenate([i.values.reshape(-1, width) for i in info.values()])
                data.fields[name] = Field(L2(data), _array(values, element, dtype))
        elif space_name in ("HGRAD", "Bezier") and basis != "F" and (
                shapes.Pyramid not in [SHAPES[t] for t in cell_types]
                or (space_name, basis, field_order) == ("HGRAD", "C", 2)):
            # Pyramids' C2 is copied (Tack's order 2 there); their other bases are
            # rational, or not Tack's: left out.
            target = max(field_order, 1)
            if shapes.Pyramid in [SHAPES[t] for t in cell_types]:
                nodes, per_cell = _copied_pyramids(grid, attribute, info, cell_types, counts,
                                                   corners, shape_infos=infos)
            else:
                nodes, per_cell = _resampled(grid, attribute, cell_types, counts, target)
            if shared and target == 1:
                space = H1(data)
                values = np.zeros((len(points), width))
                values[connectivity] = nodes
            elif shared:
                space = H1(data, order=target, basis="equispaced")
                if len(set(per_cell)) > 1:
                    left_out.append(f"{name} (shared on several cell types)")
                    continue
                values = np.zeros((space.size, width))
                values[_h1_indices(data, space, per_cell[0]).reshape(-1)] = nodes
            else:
                space = L2(data, order=target, basis="equispaced", layout="blocks") \
                    if target > 1 else L2(data)
                values = nodes
            data.fields[name] = Field(space, _array(values, element, dtype))
        else:
            left_out.append(f"{name} ({space_name} {basis}{field_order})")
    if left_out:
        warnings.warn("vtk_to_dataset left out cell grid attributes Tack has no space for: "
                      + ", ".join(left_out), stacklevel=3)
    return data


def _array(values, element, dtype):
    """Host ``values`` (``(n, width)``) as a Tack array of ``element``."""
    values = np.asarray(values, np.float64)
    out = tack.empty((len(values),), element)
    tack.copyto(out, (values.reshape(-1) if values.shape[1] == 1 else values)
                .astype(dtype.numpy_dtype))
    return out


# ── Writing ─────────────────────────────────────────────────────────

#: Tack's shapes and VTK's DG cell types.
CELL_TYPES = {shape: name for name, shape in SHAPES.items()}

_SUPERSCRIPTS = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def _space_name(width):
    """VTK's name of a ``width``-component attribute's space (``"ℝ³"``)."""
    return "ℝ" + str(width).translate(_SUPERSCRIPTS)


def _vtk_array(values, name, ids=False):
    """Host ``values`` (``(n, components)``) as a VTK array named ``name``."""
    from vtkmodules.vtkCommonCore import vtkDoubleArray, vtkIdTypeArray

    values = np.ascontiguousarray(values, np.int64 if ids else np.float64)
    out = (vtkIdTypeArray if ids else vtkDoubleArray)()
    out.SetName(name)
    out.SetNumberOfComponents(values.shape[1] if values.ndim > 1 else 1)
    out.SetNumberOfTuples(values.shape[0])
    np.asarray(out).reshape(values.shape)[...] = values
    return out


def _host(values, width):
    """A field's values on the host, ``(n, width)`` doubles."""
    from tack import arrays

    return np.asarray(arrays.to_host(arrays.materialize(values)), np.float64).reshape(-1, width)


def _nodes_written(shape, order):
    """The nodes of ``shape`` a written attribute of ``order`` holds a cell: VTK's
    order-2 pyramid's 18, else Tack's equispaced lattice's."""
    return 18 if shape is shapes.Pyramid else len(_lattice(shape, order))


def _basis_written(shape, order):
    """``(basis, order)`` an attribute of ``order`` is written in on ``shape``: VTK's C2
    on pyramids (it has no arbitrary-order pyramid), its arbitrary-order A elsewhere."""
    if shape is shapes.Pyramid:
        if order != 2:
            raise NotImplementedError(f"a pyramid of order {order} in a cell grid: VTK's "
                                      "pyramids are of order 1 or 2 (C)")
        return "C", 2
    return "A", order


def _to_vtk_order(shape, rows):
    """``rows`` (a row a cell, Tack's nodes along axis 1) in VTK's numbering: a pyramid's
    in its C2 basis' order, Tack's corners being vtkDGPyr's; other shapes' as they are."""
    if shape is not shapes.Pyramid:
        return rows
    order = _pyramid_node_order(_PYRAMID_C1)
    out = np.empty_like(rows)
    out[:, order] = rows
    return out


def to_cell_grid(data):
    """``data`` as a ``vtkCellGrid``: each shape's cells a DG cell type (in Tack's
    cell order within it), the geometry its shape attribute and each field an
    attribute -- order 1 in VTK's complete basis ``C`` (shared through the points
    for ``H1``, per cell for ``L2``), higher orders in its arbitrary-order basis
    ``A`` (lexicographic, the first axis fastest, equispaced, as Tack's
    equispaced nodes), ``Constant`` fields ``constant``. Fields of other spaces
    are left out with a warning."""
    from vtkmodules.vtkCommonDataModel import vtkCellAttribute, vtkCellGrid
    from vtkmodules.vtkFiltersCellGrid import (
        vtkDGEdge,
        vtkDGHex,
        vtkDGPyr,
        vtkDGQuad,
        vtkDGTet,
        vtkDGTri,
        vtkDGVert,
        vtkDGWdg,
    )

    from tack import arrays
    from tack.dataset import H1, L2, Constant, UnstructuredTopology, Values
    from tack.dataset.algorithms.extract_cells import extract_cells
    from tack.dataset.algorithms.project import project
    from tack.interop.mfem import _h1_indices

    _register()
    classes = {"vtkDGVert": vtkDGVert, "vtkDGEdge": vtkDGEdge, "vtkDGTri": vtkDGTri,
               "vtkDGQuad": vtkDGQuad, "vtkDGTet": vtkDGTet, "vtkDGHex": vtkDGHex,
               "vtkDGWdg": vtkDGWdg, "vtkDGPyr": vtkDGPyr}
    if not isinstance(data.topology, UnstructuredTopology):
        if not getattr(data.topology, "reference_cells", True):
            raise NotImplementedError("a cell grid of polygons, polyhedra or polylines: VTK's "
                                      "DG cells have reference elements")
        data = extract_cells(data, np.arange(data.num_cells))
    if not isinstance(data.geometry.space, H1):
        raise NotImplementedError("a cell grid's shape is shared (an H1 geometry)")
    topology = data.topology
    types, offsets, connectivity = (
        np.asarray(arrays.to_host(arrays.materialize(a))).reshape(-1)
        for a in (topology.types, topology.offsets, topology.connectivity))
    by_id = {int(shape.ID): shape for shape in SHAPES.values()}
    seen = list(dict.fromkeys(types.tolist()))           # types by first appearance
    present = [by_id[t] for t in seen if t in by_id]
    if set(types.tolist()) - set(by_id):
        raise NotImplementedError("a cell grid of shapes VTK's DG cells do not have "
                                  "(pixels and voxels are taken as quads and hexahedra)")
    cells = {shape: np.flatnonzero(types == shape.ID) for shape in present}
    grid = vtkCellGrid()
    corners = {}
    first = 0
    for shape in present:
        name = CELL_TYPES[shape]
        rows = np.stack([connectivity[offsets[c]:offsets[c] + shape.NUM_POINTS]
                         for c in cells[shape]])
        conn = _vtk_array(rows, "conn", ids=True)
        grid.GetAttributes(name).AddArray(conn)
        metadata = classes[name]()
        grid.AddCellMetadata(metadata)
        metadata.SetCellSpec(conn, first)          # its cells' ids from first
        corners[shape] = conn
        first += len(cells[shape])

    left_out = []

    def table(space, shape):
        """Each of ``shape``'s cells' nodes' places in ``space`` (lexicographic; a
        pyramid's in VTK's C2 order)."""
        per_cell = _nodes_written(shape, space.order)
        if len(present) > 1:
            raise NotImplementedError("a curved or high-order cell grid of several shapes")
        ids = _h1_indices(data, space, per_cell)[cells[shape]]
        return _to_vtk_order(shape, ids)

    def attribute(name, width, shape_attribute=False):
        out = vtkCellAttribute()
        out.Initialize(name, _space_name(width), width)
        if shape_attribute:
            grid.SetShapeAttribute(out)
        else:
            grid.AddCellAttribute(out)
        return out

    def shared(out, field, group, width, offset=None):
        space = field.space
        if space.order == 1:
            values = _host(field.values, width)[:data.num_points]
            if offset is not None:
                values = values + offset
            array = _vtk_array(values, out.GetName().Data(), ids=False)
            grid.GetAttributes(group).AddArray(array)
            for shape in present:
                out.SetCellTypeInfo(CELL_TYPES[shape], group, "HGRAD", "C", 1)
                out.SetArrayForCellTypeAndRole(CELL_TYPES[shape], "values", array)
                out.SetArrayForCellTypeAndRole(CELL_TYPES[shape], "connectivity", corners[shape])
            return
        equispaced = H1(data, order=space.order, basis="equispaced")
        if space is not equispaced:
            field = project(data, field, equispaced)
        values = _host(field.values, width)
        if offset is not None:
            values = values + offset
        group = f"{group} {space.order}"
        array = _vtk_array(values, out.GetName().Data())
        grid.GetAttributes(group).AddArray(array)
        for shape in present:
            ids = _vtk_array(table(equispaced, shape), f"{out.GetName().Data()} conn", ids=True)
            grid.GetAttributes(CELL_TYPES[shape]).AddArray(ids)
            out.SetCellTypeInfo(CELL_TYPES[shape], group, "HGRAD", *_basis_written(shape, space.order))
            out.SetArrayForCellTypeAndRole(CELL_TYPES[shape], "values", array)
            out.SetArrayForCellTypeAndRole(CELL_TYPES[shape], "connectivity", ids)

    def per_cell(out, field, width):
        space = field.space
        target = space.order
        if target > 1 and ((space.basis or "gauss_lobatto") != "equispaced"
                           or space.layout != "blocks"):
            space = L2(data, order=target, basis="equispaced", layout="blocks")
            field = project(data, field, space)
        values = _host(field.values, width)
        def count(shape):
            return shape.NUM_POINTS if target == 1 else _nodes_written(shape, target)

        starts = np.concatenate([[0], np.cumsum([count(by_id[int(t)]) for t in types])])
        for shape in present:
            n = count(shape)
            rows = np.stack([values[starts[c]:starts[c] + n] for c in cells[shape]])
            if target > 1:
                rows = _to_vtk_order(shape, rows)
            array = _vtk_array(rows.reshape(len(rows), -1), out.GetName().Data())
            grid.GetAttributes(CELL_TYPES[shape]).AddArray(array)
            out.SetCellTypeInfo(CELL_TYPES[shape], "", "HGRAD",
                                *(("C", 1) if target == 1 else _basis_written(shape, target)))
            out.SetArrayForCellTypeAndRole(CELL_TYPES[shape], "values", array)

    shape = attribute("shape", 3, shape_attribute=True)
    shared(shape, data.geometry, "coordinates", 3, offset=np.asarray(data.origin))
    from tack.interop.vtk import _once

    written = set()
    for name, field in data.fields.entries():
        _once(written, "cell grid attribute", name, data)   # one namespace in a cell grid
        space = field.space
        width = arrays.width_of(field.values) or 1
        if isinstance(space, Values) and space.on == "points":
            space, field = H1(data), field.__class__(H1(data), field.values)
        if (isinstance(space, Values) and space.on == "cells") or isinstance(space, Constant) \
                or (isinstance(space, L2) and not space.varies and space.order == 0):
            out = attribute(name, width)
            values = _host(field.values, width)
            for s in present:
                array = _vtk_array(values[cells[s]], name)
                grid.GetAttributes(CELL_TYPES[s]).AddArray(array)
                out.SetCellTypeInfo(CELL_TYPES[s], "", "constant", "C", 0)
                out.SetArrayForCellTypeAndRole(CELL_TYPES[s], "values", array)
        elif isinstance(space, H1) and space.dofs is None and space.basis in (
                None, "equispaced", "gauss_lobatto") and (space.order == 1 or len(present) == 1):
            shared(attribute(name, width), field, "point-data", width)
        elif isinstance(space, L2) and not space.varies and space.basis in (
                None, "equispaced", "gauss_lobatto"):
            per_cell(attribute(name, width), field, width)
        else:
            left_out.append(f"{name} ({space!r})")
    if left_out:
        warnings.warn("dataset_to_vtk left out fields a cell grid has no basis for here: "
                      + ", ".join(left_out), stacklevel=3)
    return grid
