### Fixed

- `clip` through mesh points oblique to the cells (a plane through a node, not
  along a layer of faces) cuts each edge from such a point at the point itself
  (#309). Before, the case tables named that crossing by its edge, as VTK's
  `vtkTableBasedClipDataSet` does: a point on top of the corner, pieces of no
  volume (some inside out) around it, and cells naming a place twice once
  `merge_points` merged them. Pieces that collapse to no volume are now dropped
  and the others become the shape they are (a tetrahedron, pyramid, wedge or
  triangle); a hexahedral piece that loses an edge is no shape and stays a
  hexahedron with that corner named twice, as VTK writes a collapsed
  hexahedron. The records for every case and set of corners at the value
  are generated (`packages/tack-vis/tools/make_clip_snap_tables.py`).
