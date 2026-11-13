### Changed

- A contour or slice of high-order data finds its exact crossings about three
  times faster. Along a sub-edge, an `H1` or `L2` field is a polynomial of known
  degree, so each crossing samples the field at that many Chebyshev points once
  and runs its root steps on their interpolant, the same polynomial, rather
  than evaluating the field at every step. The roots are the same to rounding.
  A field of another space, or of a degree above 8 along the sub-edge, is
  evaluated at every step as before. On 265k order-4 DG hexahedra the root
  finding went from 1.14 to 0.37 s for 3.4M crossings.
- A high-order `contour` refines only the fields it carries (`fields=`) and
  the one it cuts, not every field of the dataset.
