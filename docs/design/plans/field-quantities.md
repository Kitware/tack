# Field quantities

**Status: proposed.** Issue [#467](https://github.com/Kitware/tack/issues/467),
2026-11-15. Related: [#459](https://github.com/Kitware/tack/issues/459), units per
component.

## Summary

A field's unit says what its numbers are in, not what they are. Pressure and
stress share a unit, as do torque and energy, and nothing records whether a
vector is a velocity, a vorticity or three unrelated numbers. Filters that
need to know (a transform, a coarsening, a resampling) either ask the caller
or assume. This plan adds a **quantity** to `Field`, beside its unit: the kind
of quantity, its rank, whether it is intensive or extensive, and its parity.
It is optional and `None` by default, so nothing changes until a field has
one, and each filter declares what it does to a quantity, one filter at a
time.

The hooks are the part that is hard to add later: once many filters carry
fields without consulting a quantity, every one has to be revisited. The
registry of quantities and the rules can grow afterwards.

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
  joules.
- **Components that are different quantities.** #459 asks for a unit per
  component, for a conserved state `[rho, rho*u, rho*v, rho*w, E]` or a
  position with a time beside it. Those are several quantities stored
  together, which a unit per component describes only in part.

## What exists today

- **Units.** `Field(space, values, unit=None, gradient=None)`; `unit` is a
  string in unyt's canonical spelling or `None` for unknown, and filters keep
  it where they keep the quantity (`tack.dataset.units`).
- **Rank, implicitly.** A field's dtype (scalar, `vec3`, `mat3`) and its space
  (`HDiv` and `HCurl` are vectors). tack-expr's checker types every result as
  scalar, vector or tensor, and reads a symmetric tensor stored as six
  components in a fixed order (XX YY ZZ XY YZ XZ).
- **Location and orientation**, explicitly, in the space: points, cells,
  faces, a basis and its order, and oriented face values that `carry` negates
  where a face turns around.
- **Roles named by the caller**: `transform`'s `vectors` and `normals`,
  `deposit`'s `reduce` and `per_volume`.
- **Names.** tack-expr's `fx.cgns()` dictionary maps CGNS's data-name
  identifiers (`Pressure`, `Density`, `VelocityX`, ...) to units.

## Decisions

### D1. A quantity on a field

`Field(space, values, unit=None, gradient=None, quantity=None)`, with
`Field.quantity` read-only, `None` for unknown. A `td.Quantity` is a small
immutable record:

| field | meaning |
|---|---|
| `kind` | the kind of quantity, a name from the registry (D5): `"pressure"`, `"velocity"`, `"stress"` |
| `rank` | `"scalar"`, `"vector"`, `"tensor"` or `"symmetric tensor"` |
| `extent` | `"intensive"` (a density, a velocity) or `"extensive"` (a mass, a count, a flux through a face) |
| `parity` | `"polar"` or `"axial"` (a pseudovector or pseudoscalar) |

The constructor checks what it can: the unit has the kind's dimensions, and
the values' width fits the rank (1, 3, 9 or 6). The unit stays the storage
unit; a quantity never converts anything.

*Recommended.*

### D2. Each filter declares what it does to a quantity

`carry` and the filters that make new fields consult the quantity through one
rule per filter, so a filter's behaviour is written in one place:

- **carried unchanged:** extracting, clipping, thresholding cells;
- **needs intensive:** interpolation to new points, resampling, probing, AMR
  coarsening by averaging; an extensive field is summed where the filter can
  (coarsening, deposit) and refused, naming it, where it cannot;
- **needs rank and parity:** `transform` turns every vector and tensor field
  by rank, flips an axial one under a reflection, and keeps `vectors` and
  `normals` for fields with no quantity;
- **makes a new quantity:** `gradients` raises the rank and divides the
  dimensions by length; `integrate` turns a density into its extensive total;
  `magnitude` takes a vector to its scalar.

A field with no quantity is treated exactly as today. So the rules land one
filter at a time, each with its test.

*Recommended.*

### D3. Components that are different quantities

A field whose components are different quantities carries one quantity per
component (`quantity=(q0, q1, ...)`), each with its own unit, which is what
#459 asks for. Filters that need one quantity (`magnitude`, `dot`,
`gradients`, advection's velocity) refuse such a field by name; the rest
carry it component by component.

*Recommended*, as the answer to #459. Other: keep #459's unit per component
and leave quantities whole-field only.

### D4. Frames: the geometry's and a field's, later

Every vector and tensor is in the dataset's global Cartesian frame for now,
and a field in another frame (a rotating frame's relative velocity,
cylindrical components) is converted on the way in. A frame, when it comes,
is its own object: a parent, a transform to it and a kind (Cartesian,
cylindrical, spherical), so frames nest (a blade's in a rotor's in the
lab's). The geometry has one, which `DataSet.origin` and an image's direction
matrix begin; a vector or tensor field has one, the geometry's unless given.
`transform` is then a change of frame, by D2's rank and parity.

A frame is not part of a quantity, but a moving frame needs both: between a
rotating frame and the lab, a velocity gains the frame's
angular velocity crossed with the position, and a force or a stress only
turns. The kind says how a value changes; the frame says which frame it is in.

*Recommended: assume global Cartesian now, and place frames as above for a
later plan.*

### D5. The registry

Kinds of quantity come from a registry seeded from CGNS's data-name
identifiers, which tack-expr's dictionary already maps to units, each with its
dimensions, rank and extent. Users add kinds with `td.quantities.add(...)`.
ISO 80000's names are the reference where CGNS has none.

*Recommended.*

### D6. Display units: later

A unit system mapping kinds to preferred units (SI, Imperial, a user's) is a
display concern, for trame apps and colour bars, built on `units.conversion`.
It needs D1 and nothing in the filters.

*Recommended: defer.*

## Not in this plan

- **What a quantity is of** (a material, a phase) and how it is defined (a
  constant, Sutherland's law): the solver's or the naming's concern.
- **Time averaging** (mean, RMS) as a property of a quantity.
- **Provenance** (how a field was derived), beyond what an expression field
  already keeps.

## Open questions

- **tack-expr's inference.** Which expressions name a quantity: a field and
  its scaling keep it, a sum of one kind keeps it, `grad` and `mag` make one,
  and anything mixed gives `None`, as units do. Is that enough?
- **Interop.** VTK has no quantity, and its units proposal is still under
  discussion ([Units in VTK](https://discourse.vtk.org/t/units-in-vtk-proposal/16618)).
  Until it has one, a quantity could come from a CGNS name on import and go
  out as an information key.
- **A default quantity from a name.** Whether a field stored as `"Pressure"`
  should take the pressure quantity on import, or only when asked.

## Order

Each step is its own pull request, linking this plan:

1. D1, `td.Quantity`, `Field.quantity` and its checks, with D5's registry.
2. D2's rules for `transform` (rank and parity) and AMR coarsening (extent),
   the two with wrong answers today.
3. D2's rules for the other filters, a few at a time.
4. D3, components with their own quantities (#459).
5. tack-expr's inference.
6. This plan's status: done, with what changed on the way.
