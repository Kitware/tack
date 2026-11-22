### Changed

- `contour` of a field named by its points keeps each triangle corner's point
  name alone (8 bytes), and each triangle's cell and level once, and computes
  each point's position, ends and weight once from its name after the merge: it
  carried them with every corner (40 bytes). The output is the same bits. On 5M
  tetrahedra (CPU) the peak fell from 32 to 23 MiB at one value (VTK: 28) and
  from 248 to 142 MiB at ten.
