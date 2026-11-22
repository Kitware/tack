### Added

- `extract_surface` gives `L2` (DG) fields to the surface in the same space,
  each face holding its cell's trace, on every shape with a reference element
  and every basis (equispaced, Gauss-Lobatto, Gauss-Legendre, Bernstein), and
  on a 2D mesh's boundary edges; an order-0 field takes its cell's value. The
  space keeps the layout asked for, and a surface of no faces has the fields
  with no values. They were left behind, so a DG solution's surface had none of
  its fields (#418).
