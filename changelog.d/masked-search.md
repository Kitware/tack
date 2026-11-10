### Changed

- `probe`, `advect`, `streamlines`, `voxelize` and temporal advection leave
  hidden and refined cells out in the cell search itself, where they used to
  extract the other cells into an unstructured copy (`ghosts.visible`) and
  build that copy a cell locator on every call -- every AMR box below the
  finest level did. The answers are the same, bit for bit; the dataset keeps
  its topology, its cached locator and, for an image or a rectilinear grid,
  the arithmetic search. `CellLocator.find` and its search take a query's
  marks (`hidden=data.ghosts.cells`), so one locator serves datasets of any
  marks, and a locator given to `probe`, `advect` or `streamlines` is taken on
  data with hidden cells instead of refused. A repeated probe of a 128^3
  image with a tenth of its cells refined (1M points, CPU, M1 Max) took about
  190 ms and 0.30 GB; it takes 31 ms and 0.05 GB (#402).
