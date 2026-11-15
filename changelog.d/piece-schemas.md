### Fixed

- `probe`, `integrate`, `merge_pieces` and `ghost_cells` over a
  `PartitionedDataSet` compare their pieces' fields before combining them: the
  same names and element types on every piece and rank, one unit each (and one
  geometry unit for `integrate`'s measure). They took the first piece's as
  everyone's: a probe of pieces with different fields failed after its points
  were sent (over MPI, the other ranks waited forever), and values in metres
  and centimetres were summed or merged labelled metres. A mismatch is refused
  on every rank, before any exchange, as `units.common` refuses mixed units
  elsewhere; a field whose unit one piece leaves unknown comes out unknown
  (#411, found by a code review).
