### Added

- Array views: `a[i:j]` is an array of rows `i` to `j` of `a`'s first
  dimension over the same memory, as in NumPy, on every backend (CPU, CUDA,
  HIP, Level Zero, Metal, Warp). Kernels, `fill`, `tack.copyto`, reductions,
  copies and DLPack take it as any array; writes through it land in `a`, and
  count as writes to all of `a`'s memory (`DeviceBuffer.version`). A view
  keeps `a` alive and is read-only if `a` is; a step is refused. Structure-of-
  arrays arrays slice component by component. `tack.copyto` between views that
  overlap (`copyto(a[1:], a[:-1])`) reads the source whole first, as NumPy's,
  copying it to new storage.
