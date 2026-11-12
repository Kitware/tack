### Changed

- `tack.communication.Routing` tests each point only against the boxes of its
  cell on a grid over the blocks' boxes, where it tested every point against
  every box on the host, and groups the points it sends by destination in one
  sort. The destinations and the order of what arrives are the same. A probe
  of 2M points over 64 pieces in one process (CPU, M1 Max) took 3.09 s, 2.6 s
  of it in the box tests; it takes 0.71 s. An AMR hierarchy is probed through
  it too: VTK's Enzo test hierarchy, 0.68 s to 0.43 s (#358).
