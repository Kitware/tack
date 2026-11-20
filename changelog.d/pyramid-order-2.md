### Added

- Pyramids of order 2: VTK's 18-node pyramid (corners, edge middles, the base's
  centre and the triangle faces' centroids), whose functions hold every
  quadratic. `H1(data, order=2)` and `L2(data, order=2)` take meshes with
  pyramids, beside other shapes, and a geometry of order 2 curves them. An
  `H1` field keeps a value at each triangle face a pyramid has. Bounds,
  probes, gradients, integrals, `extract_surface` (with #418's `L2` traces),
  and the cuts (`contour`, `slice`, `clip`, `threshold`, `subdivide`, at
  subdivision 1 or 2, into 6 pyramids and 4 tetrahedra) apply to them (#422).
- VTK cell grids: pyramids' `HGRAD C2`, the shape's or a field's, is read node
  by node into Tack's order 2, and written back as `C2`. It was left out.
