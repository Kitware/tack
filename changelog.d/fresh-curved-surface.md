### Fixed

- `extract_surface` of a mesh whose faces had not been derived gave a curved
  geometry (`H1` above order 1) a straight surface when no field was above
  order 1: tetrahedra lost their order-2 geometry, unless an earlier filter had
  derived the faces (#421).
