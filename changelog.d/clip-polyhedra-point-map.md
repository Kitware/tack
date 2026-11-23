### Changed

- `clip` of polyhedra (a polyhedral topology, or polyhedra beside shapes) numbers
  the input points its clipped faces keep through a map of one entry per input
  point, and sorts only the crossings, as the shapes' clip now does: it named
  every kept face point and sorted them all. The output is the same bits; where
  it keeps most of the mesh, its peak fell from 3.8 to 4.5 times its output to
  2.5 to 3.4.
