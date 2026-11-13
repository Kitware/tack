### Changed

- `Advection`, `Pathlines` and `Streaklines` over pieces on one mesh (as a
  reader gives each step) run each piece's advection kernel on the particles
  its own cells hold, whatever their rank, with the velocity blended in time,
  and send them back to their rank. A step that would leave its piece, or
  reach one of its ghost cells (whose copy of the velocity may not match the
  owner's), is taken on the host, which probes every piece once a stage for
  both steps' velocities (twice when the velocity moves between the cells and
  the points): the piece it ends in goes on, or the particle has left the
  mesh and stops where it was, as before. The paths are the host's. On a
  64 x 64 x 32 grid in eight pieces on the CPU, a step takes 75-99 ms with
  100 particles (126-1121 ms on the host) and 167-222 ms with 10,000
  (183-1769 ms), the slower cases with ten substeps. Steps
  with hidden or refined cells (`Ghosts.MASKED_CELLS`) are integrated on the
  host, since the kernel would search those cells (#311).
