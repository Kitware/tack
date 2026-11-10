### Changed

- `clip`, `extract_surface`, `extract_edges`, `triangulate`, `feature_edges` and
  `glyphs` leave hidden and refined cells out in place, where they extracted
  the other cells first: a copy of every box below an AMR hierarchy's finest
  level, its positions stored, its fields copied, and its topology's faces or
  edges derived again on every call. The output is the extracted cells', the
  same cells and points; `extract_edges` and `triangulate` keep the input's
  points, as their docstrings have it. `clip`'s polyhedra and subdivided
  fields, `extract_surface` of order-2 fields, a 2D mesh or polyhedra, its
  pieces, `extract_edges` of an `L2` geometry, and `feature_edges` and
  `glyphs` of polygons or polyhedra still take the cells left (#405).
