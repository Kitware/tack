### Changed

- Point gradients at bad corners of solid cells stay bounded, which changes
  point gradients on any mesh with such corners, not only on polyhedra. A cell
  folded over (a corner's scaled Jacobian below -1e-3) gives every corner its
  gradient as a polyhedron's, Green-Gauss over its faces, rather than inverting
  its map there; a corner flat but for rounding (within 1e-3 of 0) is left out
  of the point's average, as an exactly flat one was. On AhmedML, whose 5,493
  tangled and nearly flat hexahedra gave point gradients to 2e9, the largest is
  1.7e3 (132 read as all polyhedra), median agreement 1e-4.
