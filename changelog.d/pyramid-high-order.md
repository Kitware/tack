### Added

- Pyramids of order 3 and up, `H1` and `L2`: Bergot, Cohen and Duruflé's space
  (VTK's modal `G` pyramid's), which holds every polynomial of that degree and
  contains the order-2 pyramid's, its values at the pyramid's lattice
  (equispaced; the default above order 2 on meshes with pyramids), `H1`'s shared
  by entity with tetrahedra, hexahedra and wedges. Probes, integrals, gradients,
  bounds, `project` and `jump` take it, and the cuts subdivide pyramids at any
  level: layer by layer into upright and upside-down pyramids and tetrahedra
  (#422).
- VTK cell grids: pyramids' `HGRAD G` attributes of any order, per cell, are read
  into Tack's pyramid of that order (resampled by VTK, exactly). They were left
  out.

### Fixed

- `jump` above order 1 compared two different points of a pyramid's triangle
  face (its corners' mean in each cell's collapsed coordinates), so a continuous
  field of order 2 on pyramids had jumps there.
