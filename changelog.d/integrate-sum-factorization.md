### Changed

- `integrate` integrates a field in a tensor basis (scalar or vector) on
  hexahedra and voxels by sum factorization: the cell's values contracted
  with the 1D basis one axis at a time, so a quadrature point costs as many
  multiply-adds as the basis has nodes along an axis rather than all of the
  cell's. On a curved geometry in a tensor basis the measure element is
  computed the same way: the cell's node positions read once and contracted
  with the 1D basis and its derivative, rather than every node read at every
  point. An order-4 DG field on 265k order-2 hexahedra: 1.02 s to 0.045 s a
  step for a scalar, 1.01 s to 0.064 s for a 3-vector; on Metal 0.12 s to
  0.037 s. Other cells and geometries are evaluated a point at a time as
  before.
