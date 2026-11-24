### Added

- `ghost_cells` of unstructured pieces carries fields of higher order and a
  curved geometry: `H1` fields above order 1, `L2` fields (an order per cell
  too) and a curved `H1` geometry, each ghost cell with its nodes, placed
  through the grown piece's own numbering, so a node it shares with the piece's
  cells is one value, the piece's own (as its points keep theirs). An `H1`
  field numbered by an element-to-DOF table comes numbered by entity, as
  `merge_pieces` gives it. They were refused. Structured pieces carry them too,
  remade on the grown blocks with the ghosts' values from their owners (#328).
- `update_ghosts` refreshes fields of higher order on pieces `ghost_cells` made
  (`H1` above order 1, `L2`, an order per cell too): each value a piece holds
  but does not own from its owner, the lowest piece with a cell of its own that
  has it, so a node on an edge or face pieces share is owned with that edge's or
  face's points and the field stays one value along it. It refused them (#328).
