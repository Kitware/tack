### Added

- `L2` of order 3 and up on pyramids: Bergot, Cohen and Duruflé's space (VTK's
  modal `G` pyramid's), which holds every polynomial of that degree and contains
  the order-2 pyramid's, its values at the pyramid's lattice (equispaced; the
  default for `L2` above order 2 on meshes with pyramids). Probes, integrals,
  gradients, bounds and `project` take it; the cuts take it with
  `subdivision=2` (#422).
- VTK cell grids: pyramids' `HGRAD G` attributes of any order, per cell, are read
  into Tack's pyramid of that order (resampled by VTK, exactly). They were left
  out.
