### Fixed

- tack-expr keeps a field's unit through a component written `v_X` (as through `v[0]`), and
  gives the coordinates (`coords`, `coordsX`, `coords[0]`) the dataset's unit; both came out
  unknown unless something else in the expression had a unit.
