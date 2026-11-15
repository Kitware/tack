### Fixed

- Probing (and advecting through) a field above order 1 on a mesh of several
  cell shapes read each cell's value first through the view standing in for
  tetrahedra, at the cell's index in its own shape's group, past the end of
  that view's group when it was smaller (9 hexahedra read at index 32): the
  value was discarded, but the read was out of bounds, a bus error on a large
  enough mesh. Each cell is now read through its own shape's view only.
