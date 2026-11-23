### Added

- `Field(..., kind=)` and `tack.dataset.Kind`: what a field's values are,
  beside their unit (the quantity's name, rank, extent, parity and variance),
  each part optional. A name fills the rest from `tack.dataset.kinds`' registry,
  seeded from CGNS's data names, and a field checks its kind against its unit's
  dimensions, its values' rank and, in `HDiv` or `HCurl`, its variance.
  `to_device`, `convert`, `project` and every filter that carries fields
  (`carry`: extracting, clipping, thresholding, shrinking) keep it; no filter
  reads it yet (#467).
- `units.same_dimensions(a, b)`: whether two units have the same dimensions.
