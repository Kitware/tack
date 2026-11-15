### Changed

- `extract_surface` keeps the boundary it finds from a mesh's derived faces (a
  structured grid's, or an unstructured mesh's once its faces are derived) with
  the topology, as it keeps the one it finds from the cells alone, so a repeated
  call on the same mesh gives the same surface geometry and its normals are
  computed once.
