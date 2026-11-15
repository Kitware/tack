### Changed

- `tack.arrays.Permutation(ids, base)` refuses ids that are not integers (or
  are vectors), with a `TypeError` naming the argument order: given the other
  way round, a base's floats were used as ids and read past the end (#376).
  The other implicit arrays refuse a float where a size, start, stride,
  offset, component or image size is meant, rather than truncating it with
  `int()`; NumPy integers serve as Python's do.
