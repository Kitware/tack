### Changed

- `clip` counts and writes its cells in batches, as VTK's `vtkBatch` does: a mesh
  of one shape keeps nothing per cell (a batch is a thread's 256 cells, or with
  workgroups a workgroup's), and a mesh of several shapes, or a structured one,
  keeps 15 bytes a cell where it kept 40 (each cell's counts and starts of five
  kinds). The output is the same bits. On 5M cells (CPU) the counted peak fell
  from 256 to 80 MiB on tetrahedra and from 315 to 194 MiB on hexahedra, below
  VTK's 111 and 210, and a warm clip from 30 to 15 ms and from 54 to 42 ms.
