### Fixed

- A negative `tack.constant` negated in a kernel (`-LOWEST`, with
  `LOWEST = tack.constant(-3.0e38)`) compiles on the GPU backends: their
  generators wrote `--3e+38f`, a decrement of a literal, which Metal's
  compiler (and any C compiler) refuses. The CPU was not affected.
