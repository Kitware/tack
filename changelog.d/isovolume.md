### Added

- `isovolume(data, by, lower, upper)`: the part of the cells where a point
  field or an implicit function lies in a range, cut at both ends (ParaView's
  Iso Volume), and its VTK algorithm `Isovolume`. A curved geometry or a field
  above order 1 is cut through subdivision, as `clip` cuts it, into
  `PartitionedDataSet([whole, part])`, and the crossings at both ends are at
  the field's roots: the second end's are found along the sub-cells' edges the
  first clip left (#419).
