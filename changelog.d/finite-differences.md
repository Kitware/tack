### Added

- `tack.dataset.finite_differences` (#315): the gradient (`derivative=1`) or the
  second derivatives (`derivative=2`, the Hessian, 9 values per component) of a
  point or cell field on an image, a rectilinear or a curvilinear grid, by
  finite differences in index space taken to world axes through the grid's own
  differences. `order` is 2, 4 or 6; at order 2 the gradient is VTK's
  vtkGradientFilter's on structured data, first order at the edges as VTK's is,
  and orders 4 and 6 keep their order at the edges with one-sided stencils. A
  separate filter from `gradients`, which stays the one for finite element
  data; `flow_quantities` takes either.
