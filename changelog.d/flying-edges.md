### Changed

- `contour` of a structured grid of solid cells -- an image, a rectilinear or a
  curvilinear grid -- with a point field goes by flying edges, as VTK's
  `vtkFlyingEdges3D`: crossings counted along the grid's rows, numbered by a
  scan and written once, with no per-corner arrays and no sort. The triangles,
  points and carried values are the general path's, bit for bit, up to their
  order. On an 11M-cell image at ten values (CPU, M1 Max) the peak above the
  input fell from 2.24 to 0.32 GB and the time from 259 to 219 ms; on an
  80M-cell image at two values, from 1.32 to 0.30 GB and 346 to 263 ms (#316).
- `slice` of an image or a rectilinear grid by a `Plane` (`slice_plane`) goes
  by a plane cutter, as VTK's `vtkFlyingEdgesPlaneCutter`: flying edges over
  each row's stretch near the plane, found from the plane's equation, the
  distance computed where it is read. Only the cells near the plane are
  visited and nothing is stored for the others; the output is the general
  path's, bit for bit, up to its order. A plane through a 432^3 image (80M
  cells, CPU, M1 Max) took 177 ms and 0.71 GB above the input; it takes 14 ms
  and 0.03 GB.
