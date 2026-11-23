### Fixed

- An array held by more than one device-class object (or by one object under
  two names) is one kernel parameter. Objects over the same space hold the
  same arrays, so tack-expr's divergence of a tensor on a quadratic grid
  passed 289 pointers, past Level Zero's 2 KB of kernel arguments; it now
  passes 35 (#474).
