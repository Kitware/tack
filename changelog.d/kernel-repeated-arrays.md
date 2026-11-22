### Fixed

- A kernel given one array through several parameters takes it once, through
  the first. Device-class objects over the same space hold the same arrays,
  so tack-expr's divergence of a tensor on a quadratic grid passed 289
  pointers, past Level Zero's 2 KB of kernel arguments; it now passes 35
  (#474).
