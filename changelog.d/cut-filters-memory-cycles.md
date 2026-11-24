### Fixed

- A call that compiled a new variant kept its arrays until Python's next
  cycle collection: the IR traversals' recursive closures (`transform_ir`,
  `substitute_ir`) formed a reference cycle holding variant resolution's
  rewrite, and through it the call's arrays. They are now given back when
  dropped; a hexahedra contour at 5M cells peaked 14 MiB lower on the CPU
  (#318).
- The bucket sort under contour, slice and the point merges (`bucket_order`,
  `sorted_key_order`) reuses its counts as the scatter's fill, one array as
  long as the buckets fewer: 19 MiB less for a hexahedra contour or slice at
  5M points on CUDA, HIP and Level Zero (#318).
