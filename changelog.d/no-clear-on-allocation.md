### Changed

- CUDA, HIP and Level Zero no longer clear every new array, which nothing
  promised (`tack.empty`'s contents are unspecified on every backend): each
  temporary was written once more than needed and `tack.zeros` cleared twice.
  Code that read an array before writing it, and worked on those GPUs by
  reading zeros, now reads whatever the allocator left (#386).

### Added

- `TACK_FILL_NEW=0xFF` (any byte) fills every new array with that byte on every
  backend, so a read before a write reads NaN floats, -1 signed integers and an
  unsigned type's largest value instead of zeros that hide it; debug mode fills
  with 0xFF unless it gives another byte or `off` (#386).

### Fixed

- CUDA: an exportable array (`tack.empty(..., exportable=True)`, or the copy
  `export_memory` makes of any other) dropped while a launch, fill or copy on it
  was still queued faulted the context (`CUDA_ERROR_ILLEGAL_ADDRESS`); its
  memory is now released once Tack's stream is idle. And the first
  `export_memory` of an array a whole number of granules long handed out its
  handle while the copy into exportable memory could still be running; it now
  waits for it (#386).
