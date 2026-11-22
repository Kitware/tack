### Changed

- Locating a point in a polyhedron is cheaper: its sub-tetrahedra are first
  tested only for holding the point, and the one nearest it is sought only where
  none does and the caller can use it. A particle's check of the cell it was
  last in never can, and the search of a locator's bins stops seeking it once a
  cell holds the point. Results are unchanged, bit for bit. Streamlines on a
  4M-cell mesh of polyhedra take 113 ns a path point on the CPU (171 before) and
  about 260 on Metal (320); with every tenth cell of a hexahedral mesh a
  polyhedron, 98 (103) and 163 (186) (#451).
