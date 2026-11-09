### Changed

- `contour` of a structured grid of solid cells -- an image, a rectilinear or a
  curvilinear grid -- with a point field goes by flying edges, as VTK's
  `vtkFlyingEdges3D`: crossings counted along the grid's rows, numbered by a
  scan and written once, with no per-corner arrays and no sort. The triangles,
  points and carried values are the general path's, bit for bit, up to their
  order. On an 11M-cell image at ten values (CPU, M1 Max) the peak above the
  input fell from 2.24 to 0.32 GB and the time from 259 to 219 ms; on an
  80M-cell image at two values, from 1.32 to 0.30 GB and 346 to 263 ms (#316).
