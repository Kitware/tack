# Field quantities

**Status: proposed.** Issue [#467](https://github.com/Kitware/tack/issues/467),
2026-11-15. Related: [#459](https://github.com/Kitware/tack/issues/459), units per
component.

## Summary

A field's unit says what its numbers are in, not what they are. Pressure and
stress share a unit, as do torque and energy, and nothing records whether a
vector is a velocity, a vorticity or three unrelated numbers. Filters that
need to know (a transform, a coarsening, a resampling) either ask the caller
or assume. This plan adds a **kind** to `Field`, beside its unit: the name of
the quantity, its rank, whether it is intensive or extensive, its parity, and
how it transforms. Every part is optional and `None` by default, so nothing
changes until a field has one, and each filter declares what it does to a
kind, one filter at a time.

The hooks are the part that is hard to add later: once many filters carry
fields without consulting a kind, every one has to be revisited. The
registry of kinds and the rules can grow afterwards.

## Why

Each of these runs and gives a wrong answer silently, or works only because
the caller knew to say something:

- **A rotation leaves a velocity behind.** `td.transform(data, matrix)` turns
  only the fields named in `vectors` and `normals` ("Tack keeps no roles, so
  here they are named"). A velocity not named stays in the old frame.
- **A reflection turns a vorticity the wrong way.** Naming a vorticity in
  `vectors` under a matrix with a negative determinant maps it as a polar
  vector. Vorticity, angular velocity and a magnetic field are pseudovectors:
  under a reflection they take the opposite sign.
- **Averaging an extensive quantity.** AMR gives a refined coarse cell the
  volume-weighted average of the fine cells over it, which is right for a
  density and wrong for a per-cell mass or a count. `deposit` leaves the
  choice to the caller (`reduce`, `per_volume`).
- **One unit, two quantities.** `units.named` shows a derived unit by its SI
  name from dimensions alone, so a torque in `N*m` and an energy in `J` cannot
  be told apart, and a display unit chosen by dimensions would show torque in
  joules. A kind makes the two distinguishable; showing each in its own unit
  is D6, which this plan defers.
- **Components that are different quantities.** #459 asks for a unit per
  component, for a conserved state `[rho, rho*u, rho*v, rho*w, E]` or a
  position with a time beside it. Those are several quantities stored
  together, which a unit per component describes only in part.

## What exists today

- **Units.** `Field(space, values, unit=None, gradient=None)`; `unit` is a
  string in unyt's canonical spelling or `None` for unknown, and filters keep
  it where they keep the quantity (`tack.dataset.units`). `units.quantity`
  makes a magnitude with a unit, which is why this plan's record is a kind
  and not a quantity.
- **Rank, implicitly.** A field's dtype (scalar, `vec3`, `mat3`) and its space
  (`HDiv` and `HCurl` are vectors, stored as one value per degree of
  freedom). tack-expr's checker types every result as scalar, vector or
  tensor, and reads a symmetric tensor stored as six components in VTK's
  order (XX YY ZZ XY YZ XZ).
- **Location and orientation**, explicitly, in the space: points, cells,
  faces, a basis and its order, and oriented face values that `carry` negates
  where a face turns around.
- **Roles named by the caller**: `transform`'s `vectors` and `normals`,
  `deposit`'s `reduce` and `per_volume`. `transform` takes any affine map,
  a shear too, and gives `normals` the inverse transpose of its linear part.
- **Names.** tack-expr's `fx.cgns()` dictionary maps CGNS's data-name
  identifiers (`Pressure`, `Density`, `VelocityX`, ...) to units.

## Decisions

### D1. A kind on a field

`Field(space, values, unit=None, gradient=None, kind=None)`, with
`Field.kind` read-only, `None` for unknown. A `td.Kind` is a small immutable
record, each part optional:

| part | meaning |
|---|---|
| `name` | the quantity, a name from the registry (D5): `"pressure"`, `"velocity"`, `"stress"` |
| `rank` | `"scalar"`, `"vector"`, `"tensor"` or `"symmetric tensor"` |
| `extent` | what the field evaluates to: `"intensive"` (a density, a velocity) or `"extensive"` (a mass, a count) |
| `parity` | `"polar"` or `"axial"` (a pseudovector or pseudoscalar) |
| `variance` | how it turns under a linear map: `"contravariant"` (a velocity), `"covariant"` (a gradient, a normal) or, for a tensor, `"mixed"` (a velocity gradient) |

Given a `name`, the registry fills the parts not given, and the constructor
checks that given parts agree with it. Without one, each part stands alone:
a field read from VTK's active vectors has a rank and nothing else, and a
user may know only that a field is extensive.

Extent is about what the field evaluates to, not how it is stored. A
first-order `HDiv` velocity stores each face's integrated flux, extensive
numbers, and is still an intensive field; the space already says what its
values are.

The constructor checks what it can: the unit has the name's dimensions, and
the rank fits the space's value shape (not the values' width, which for
`HDiv` and `HCurl` is one per degree of freedom, and for a 2D vector is 2).
A symmetric tensor's six components are in VTK's order, XX YY ZZ XY YZ XZ,
as tack-expr reads them. The unit stays the storage unit; a kind never
converts anything.

*Recommended.*

### D2. Each filter declares what it does to a kind

`carry` and the filters that make new fields consult the kind through one
rule per filter, so a filter's behaviour is written in one place:

- **carried unchanged:** extracting, clipping, thresholding cells;
- **needs intensive:** interpolation to new points, resampling, probing, AMR
  coarsening by averaging. An extensive field is summed where the filter can
  (coarsening, deposit). Where it cannot, a field named in the call is
  refused, naming it, and a call that carries every field leaves the
  extensive one out and warns, naming it;
- **needs rank, parity and variance:** `transform` turns each vector and
  tensor field by the linear part `M` of its map: a contravariant vector by
  `M v`, a covariant one by `M⁻ᵀ g`, an axial one by `det(M) M⁻ᵀ ω`, and a
  tensor by both its indices (a velocity gradient `M G M⁻¹`, a stress
  `M σ Mᵀ`). For a rotation or a reflection these reduce to rank and parity,
  and that is the first step (Order); until variance lands, a non-orthogonal
  map refuses a field with a kind, naming it. A field with a kind that is
  also named in `vectors` or `normals` is refused, as is a scalar named
  there. Fields with no kind keep `vectors` and `normals` as today;
- **makes a new kind:** `gradients` raises the rank, divides the dimensions
  by length and adds a covariant index; `integrate` turns a density into the
  total the registry names for it (D5), and gives `None` for a kind with no
  total, a temperature or a velocity; `magnitude` takes a vector to its
  scalar.

A field with no kind is treated exactly as today. So the rules land one
filter at a time, each with its test.

*Recommended.*

### D3. Components that are different quantities

A field whose components are different quantities is split into one field
per quantity, as views at no copy: `ExtractComponent` for one component, and
`Components` over three of them for a vector. `[rho, rho*u, rho*v, rho*w, E]`
becomes a density, a momentum vector (components 1 to 3, which `transform`
turns together) and an energy, each with its own kind and unit. Every field
then has one kind and one unit, no filter needs a mixed path, and
`Field.unit` keeps one meaning.

*Recommended*, as the answer to #459. Other: a kind per group of components
(`((0, density), (1..3, momentum), (4, energy))`), which keeps the stored
field whole and makes every filter handle groups.

### D4. Frames: the geometry's and a field's, later

Every vector and tensor is in the dataset's global Cartesian frame for now,
and a field in another frame (a rotating frame's relative velocity,
cylindrical components) is converted on the way in. A frame, when it comes,
is its own object: a parent, a transform to it and a kind of frame
(Cartesian, cylindrical, spherical), so frames nest (a blade's in a rotor's
in the lab's). The geometry has one, which `DataSet.origin` and an image's
direction matrix begin; a vector or tensor field has one, the geometry's
unless given. `transform` is then a change of frame, by D2's rules.

A frame is not part of a kind, but a moving frame needs both: between a
rotating frame and the lab, a velocity gains the frame's angular velocity
crossed with the position, and a force or a stress only turns. The kind says
how a value changes; the frame says which frame it is in.

*Recommended: assume global Cartesian now, and place frames as above for a
later plan.*

### D5. The registry

Kinds come from a registry seeded from CGNS's data-name identifiers, which
tack-expr's dictionary already maps to units. Each entry has its dimensions,
rank, extent, parity and variance, and, for a density, the kind it is a
density of (a mass density of a mass), which `integrate` uses. Kinds compare
by value, and a check across pieces and ranks gives the one kind of several
or names the two that differ, as `units.common` does for units. Users add
kinds with `td.kinds.add(...)`. ISO 80000's names are the reference where
CGNS has none.

*Recommended.*

### D6. Display units: later

A unit system mapping kinds to preferred units (SI, Imperial, a user's) is a
display concern, for trame apps and colour bars, built on `units.conversion`.
It needs D1 and nothing in the filters.

*Recommended: defer.*

## Not in this plan

- **What a quantity is of** (a material, a phase) and how it is defined (a
  constant, Sutherland's law): the solver's or the naming's concern.
- **Time averaging** (mean, RMS) as a property of a kind.
- **Provenance** (how a field was derived), beyond what an expression field
  already keeps.

## Open questions

- **tack-expr's inference.** Which expressions name a kind: a field and its
  scaling keep it, a sum of one kind keeps it, `grad` and `mag` make one,
  and anything mixed gives `None`, as units do. Is that enough?
- **Interop.** VTK has no kind, and its units proposal is still under
  discussion ([Units in VTK](https://discourse.vtk.org/t/units-in-vtk-proposal/16618)).
  VTK's attribute roles can map both ways without one: active vectors to a
  vector with no name, normals to a covariant vector, tensors to a tensor,
  and texture coordinates and global ids to no kind. Beyond that a kind
  could come from a CGNS name on import and go out as an information key.
- **A default kind from a name.** Whether a field stored as `"Pressure"`
  should take the pressure kind on import, or only when asked.

## Order

Each step is its own pull request, linking this plan:

1. D1, `td.Kind`, `Field.kind` and its checks, with D5's registry.
2. D2's rule for AMR coarsening (extent), a wrong answer today that needs
   nothing else.
3. D2's rule for `transform` under rotations and reflections (rank and
   parity), the other wrong answer today, refusing other maps for fields
   with a kind.
4. D2's rules for the other filters, a few at a time.
5. `transform` under any affine map (variance).
6. D3, splitting a field into one per quantity (#459).
7. tack-expr's inference.
8. This plan's status: done, with what changed on the way.
