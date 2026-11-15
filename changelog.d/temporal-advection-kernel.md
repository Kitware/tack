### Changed

- `Advection`, `Pathlines` and `Streaklines` integrate two steps on one mesh
  in one launch of the advection kernel `advect` and `streamlines` use, with
  the velocity blended in time between the steps, on the steps' device. On
  the host they probed both steps' meshes at every stage, about ten probes a
  substep. One mesh is the same topology, or a new one of the same arrays and
  positions, as a reader gives each step. On a 64 x 64 x 32 grid on the CPU,
  a step takes 1.4-1.9 ms with 100 particles (14-52 ms before) and 3.6-12 ms
  with 10,000 (13-121 ms before), the slower cases with ten substeps. The
  paths are the host's: a particle whose step would leave the mesh stops
  where it was, and one reaching the step's time pauses there. Steps on
  different meshes, and pieces, are still integrated on the host (#311).
