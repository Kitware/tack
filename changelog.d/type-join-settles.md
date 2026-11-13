### Fixed

- An f64 kernel could compute part of an expression in f32: the join of the
  locals' types gave up after eight rounds, passing a widened type one copy a
  round, so the last copies of a long chain (the temporaries of inlined
  functions, a value of literals alone passed through several of them) kept
  stale f32 types and their f64 values were truncated through them. A round
  now passes a widened type on at once, the join runs until it settles, and
  not settling is an internal error. Among Tack's own kernels, locating points
  in curved cells (`probe`, advection) computed parametric points in f32 in f64
  data (#427).
