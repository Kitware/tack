### Changed

- `advect` and `streamlines` over a `PartitionedDataSet` take the velocity as an
  object whose `on(piece)` gives each piece's field (a field expression), as the
  reductions do, resolved once a piece.
