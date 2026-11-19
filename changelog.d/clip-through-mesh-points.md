### Fixed

- `clip` through a plane of mesh points keeps whole cells on one side and
  drops those on the other. Before, round-off in the nodes (a symmetry plane's,
  placed 1e-16 to either side of it by `cos(pi/2)`) had the case tables cut
  cells through a corner a hair on the dropped side: pieces of no volume, a
  diagonal split of the cell, and points twice, as VTK's
  `vtkTableBasedClipDataSet` makes them. A corner within a millionth of its
  cell's span of the value is now at the value (above it on either side, so the
  two sides of a clip fill each cell between them), the crossings on its edges
  are the corner itself, and a cell that only touches the value from the dropped
  side is dropped, as the polyhedral clip already did. The higher-order clip's
  cut cells, clipped this way, follow.
