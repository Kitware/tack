### Added

- `ghost_cells` of unstructured pieces carries fields of higher order and a
  curved geometry: `H1` fields above order 1, `L2` fields (an order per cell
  too) and a curved `H1` geometry, each ghost cell with its nodes, placed
  through the grown piece's own numbering, so a node it shares with the piece's
  cells is one value. They were refused. Structured pieces still carry point
  and cell fields only, and `update_ghosts` refreshes point and cell fields
  (#328).
