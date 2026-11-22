### Added

- `tack-expr` 0.1.0, a package of field expressions: VTK calculator syntax over
  `tack.dataset` fields, compiled to Tack kernels, with derivatives, units, a
  field dictionary, tensors and complex values. `tack_expr.field(text)` is a
  field argument for any algorithm, read on each dataset or piece; an
  expression names `normals`, `measures` and `centroids` where the dataset
  stores no field of the name.
- `tack.interop.vtk.algorithms.Calculator`: an expression in `vtkArrayCalculator`'s
  syntax as a VTK algorithm, at the points or the cells, through `tack-expr`.
