### Fixed

- `tetrahedralize` of a structured grid with hidden or refined cells splits its
  other hexahedra in 5, alternating by parity, as without them; it extracted
  the other cells first, into unstructured hexahedra it split in 6, so every
  AMR box below the finest level split unlike the finest ones. Hidden and
  refined cells make no tetrahedra, left out in place, and the points stay
  all of the input's, as the docstring has it (#403).

### Changed

- `integrate` no longer extracts the cells that are not hidden or refined
  before summing: `owned_values` already leaves hidden, refined and duplicate
  cells out, so the sums are the same, bit for bit, without the copy and with
  the dataset's cell geometry kept (#404).
