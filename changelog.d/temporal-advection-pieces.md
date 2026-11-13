### Changed

- `Advection`, `Pathlines` and `Streaklines` over pieces on one mesh (as a
  reader gives each step) run each piece's advection kernel on the particles
  its cells hold, whatever their rank, with the velocity blended in time, and
  send them back to their rank. A step that would leave its piece is taken on
  the host, which probes every piece once a stage for both steps' velocities:
  the piece it ends in goes on, or the particle has left the mesh and stops
  where it was, as before. The paths are the host's, with ghost cells or
  without. On a 64 x 64 x 32 grid in eight pieces on the CPU, a step takes
  75-94 ms with 100 particles (122-1088 ms on the host) and 164-204 ms with
  10,000 (178-1654 ms), the slower cases with ten substeps; with two layers of
  ghost cells, fewer steps leave a piece: 61-62 ms and 95-131 ms. Steps with
  hidden or refined cells (`Ghosts.MASKED_CELLS`) are integrated on the host,
  since the kernel would search those cells (#311).
