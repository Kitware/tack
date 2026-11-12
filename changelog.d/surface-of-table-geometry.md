### Fixed

- `extract_surface` of a mesh whose order-2 geometry or field is numbered by an
  element-to-DOF table (`H1(order=2, dofs=...)`, as MFEM and VTK's cell grids store
  them) gathered the surface's values as if numbered by entity: wrong values, or a
  read past the end of the array (a crash on a large mesh). Such a field is renumbered
  first.
