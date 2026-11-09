### Fixed

- Over MPI, an algorithm over pieces that refuses its pieces now raises on
  every rank. Before, only the ranks whose pieces failed the check raised, and
  the rest (ranks holding no pieces among them) waited forever in the next
  collective call. Affected: a missing field, or one of the wrong kind, width
  or space, a geometry or cells an algorithm does not take, pieces already
  ghosted. The checks of `advect`, `streamlines`, `probe`, `deposit`,
  `ghost_cells`, `update_ghosts`, `ghost_points`, `merge_pieces`,
  `extract_surface`, `connected_components`, `generate_global_ids`,
  `halo_catalog`, `statistics`, `entropy`, `integrate`, `voxel_downsample`,
  temporal advection, and the AMR hierarchy's `ghost_cells`, `update_ghosts`,
  `average_down` and constructor now run before the first collective, and
  every rank learns their outcome in a gather the algorithm already makes, or
  one more. The rank that found the error raises it; the others raise one of
  the same kind and message, with a note naming that rank (#364).
