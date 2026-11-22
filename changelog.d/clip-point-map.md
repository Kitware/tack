### Changed

- `clip` numbers the input points its cells keep through a map of one entry per
  input point, and sorts only the crossings on cut edges, as VTK's
  `vtkTableBasedClipDataSet` does: it made a record for every kept corner and
  sorted them all. The output is the same bits. On 5M cells (CPU) its peak fell
  from 364 to 256 MiB on tetrahedra and from 574 to 315 MiB on hexahedra, and a
  warm clip from 43 to 30 ms and from 61 to 54 ms.
