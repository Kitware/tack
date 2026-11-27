### Fixed

- A NaN or infinite float constant in a kernel (`tack.constant(math.nan)`,
  `tack.constant(-math.inf, tack.f64)`) compiles on the GPU backends and Warp:
  the C-family generators wrote `nanf` and `inff` (and Warp's `nan`, `inf`),
  names no dialect declares. They now spell the value's bits, reinterpreted as
  each dialect does (`__int_as_float` on CUDA and HIP, `as_float` on OpenCL,
  `as_type<float>` on Metal, `wp.nan` / `wp.inf` on Warp). The CPU was not
  affected.
