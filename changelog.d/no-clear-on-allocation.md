### Changed

- CUDA, HIP and Level Zero no longer clear every new array, which nothing
  promised (`tack.empty`'s contents are unspecified on every backend): each
  temporary was written once more than needed and `tack.zeros` cleared twice.
  Code that read an array before writing it, and worked on those GPUs by
  reading zeros, now reads whatever the allocator left (#386).

### Added

- `TACK_FILL_NEW=0xFF` (any byte) fills every new array with that byte on every
  backend, so a read before a write reads NaN floats and -1 integers instead of
  zeros that hide it; debug mode fills with 0xFF unless it is set (#386).
