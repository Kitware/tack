# Compute with expressions

A quantity that no filter computes, such as a speed, a gauge pressure or a
vorticity weighted by temperature, is a formula over fields. VTK and ParaView
write one with `vtkArrayCalculator`, the Calculator filter. Tack's counterpart
is a package of its own, `tack-expr`, imported as `tack_expr`. An expression
is text in the calculator's syntax, or the same expression written with
Python's operators on fields. `tack_expr` checks its types and units and
compiles it to one Tack kernel, which runs wherever the dataset lives.

`tack-expr` has its own version (0.1.0) and depends on `tack-vis`. In a clone
of the repository, `uv sync` installs it with the rest of the workspace.

## Evaluate an expression

`fx.evaluate(text, data, at=...)` returns a field. `at="points"` gives one
value per point and `at="cells"`, the default, one per cell, at the cell's
centre:

<!-- tack-check: evaluate -->
```python
import numpy as np

import tack
import tack.dataset as td
import tack_expr as fx

tack.set_default_device("cpu")

data = td.uniform_grid((11, 11, 11), origin=(-1, -1, -1), spacing=(0.2, 0.2, 0.2), unit="m")
x = data.positions()
rotation = np.stack([-x[:, 1], x[:, 0], np.full(len(x), 0.5, np.float32)], axis=1)
data.fields["velocity"] = td.Field(td.H1(data), tack.array(rotation, dtype=tack.vec3(tack.f32)),
                                   unit="m/s")
data.fields["T"] = td.Field(td.H1(data), tack.array(300 + 10 * x[:, 0]), unit="K")

speed = fx.evaluate("mag(velocity)", data, at="points")        # a td.Field, one value a point
hot = fx.evaluate("T > 305 ? mag(velocity) : 0", data, at="points")
vorticity = fx.evaluate("curl(velocity)", data)                # at each cell's centre

print(speed.unit, vorticity.unit)                              # m/s 1/s
assert np.allclose(speed.values.numpy(), np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2 + 0.25), atol=1e-5)
assert np.allclose(hot.values.numpy(), np.where(x[:, 0] > 0.5, speed.values.numpy(), 0))
assert np.allclose(vorticity.values.numpy(), [0, 0, 2], atol=1e-5)
```

The syntax is `vtkArrayCalculator`'s:
- **Arithmetic:** `+ - * / ^`, comparisons, `&&`, `||` and `!`.
- **Conditionals:** `a ? b : c`, or `if(a, b, c)`.
- **Components:** `v[0]` or `v_X`.
- **Functions:** `sqrt`, `exp`, `log`, the trigonometric functions, `abs`,
  `min`, `max`, `clamp` and the rest.
- **Vectors:** `mag`, `mag2`, `norm`, `dot`, `cross`, and `coords` for the
  position.

The parser checks that scalars and vectors meet where they should and that
every field can be read where the expression runs. The error points at the
column it refuses.

The expression becomes the Python source of a `@tack.kernel`, one assignment
per operation. That source is cached by its hash in `~/.cache/tack_expr`
(`TACK_EXPR_CACHE` moves it). Numbers are the kernel's arguments, so
expressions that differ only in their constants share a kernel. Text you type
never reaches the source.

## Write expressions in Python

The same expressions can be written without strings. Importing `tack_expr`
gives `td.Field`s Python's operators, so `T > 305` or `U * 2` is an
expression: a tree of operations, not yet computed. NumPy's ufuncs
(`np.sqrt`, `np.sin`, `np.arctan2`, `np.minimum`, ...) take fields and
expressions too.

<!-- tack-check: python-operators -->
```python
import numpy as np

import tack
import tack.dataset as td
import tack_expr as fx

tack.set_default_device("cpu")

data = td.uniform_grid((17, 17, 17), origin=(-1, -1, -1), spacing=(0.125, 0.125, 0.125), unit="m")
x = data.positions()
rotation = np.stack([-x[:, 1], x[:, 0], np.full(len(x), 0.5, np.float32)], axis=1)
data.fields["velocity"] = td.Field(td.H1(data), tack.array(rotation, dtype=tack.vec3(tack.f32)),
                                   unit="m/s")
data.fields["T"] = td.Field(td.H1(data), tack.array(300 + 10 * x[:, 0]), unit="K")
data.fields["wind"] = td.Field(td.Constant(data), tack.array(
    np.tile([36.0, 0.0, 0.0], (data.num_cells, 1)), dtype=tack.vec3(tack.f32)), unit="km/hr")
U, V, T = data.fields["velocity"], data.fields["wind"], data.fields["T"]

total = U + V                                            # an expression, not yet computed
relative = total.eval(data)                              # m/s: 36 km/hr is 10 m/s
centres = fx.evaluate("coords", data).values.numpy()
assert relative.unit == "m/s"
assert np.allclose(relative.values.numpy(), np.stack([10 - centres[:, 1], centres[:, 0],
                                                      np.full(len(centres), 0.5)], axis=1),
                   atol=1e-4)

speed = fx.mag(U)
warm = fx.where(T > 305, speed, 0)                       # comparisons and choices
angle = np.arctan2(U[1], U[0])                           # NumPy's ufuncs, components
print(warm.render(data)[0])                              # the calculator text it compiles to

at_points = warm.eval(data, at="points")                 # a td.Field, as fx.evaluate gives
assert np.allclose(at_points.values.numpy(),
                   np.where(x[:, 0] > 0.5, np.linalg.norm(rotation, axis=1), 0), atol=1e-5)
assert np.allclose(angle.eval(data, at="points").values.numpy(),
                   np.arctan2(x[:, 0], -x[:, 1]), atol=1e-5)

q = fx.qcrit(U).compile(data)                            # compile once, call as U changes
assert np.allclose(q(data).values.numpy(), 1, atol=1e-4)

shell = td.contour(data, speed.field(data), 1.0)         # computed where the contour reads it
assert shell.num_cells > 0

U2 = fx.ref("velocity")                                  # whatever data.fields holds then
derived = fx.compile_many({"vorticity": fx.mag(fx.curl(U2)), "helicity": fx.dot(U2, fx.curl(U2))},
                          data)(data)
assert np.allclose(derived["helicity"].values.numpy(), 1, atol=1e-4)
```

An expression is computed only when one of these methods is called:

| Method | Gives |
|---|---|
| `expr.eval(data, at=...)` | a field, as `fx.evaluate` gives |
| `expr.compile(data, at=...)` | a compiled kernel, called on the dataset as often as its fields change |
| `expr.field(data)` | a field computed wherever an algorithm reads it, as `fx.field` gives |
| `expr.function(data)` | a device class for a kernel of your own, as `fx.function` gives |

`fx.compile_many` takes expressions and text alike.

What the operators cover:
- **Arithmetic and comparisons:** `+ - * / ** %`, unary `-`, `abs()`, and
  `< <= > >=`.
- **Components:** `U[0]` of a field, and `.x`, `.y`, `.z` of an expression.
- **Conditions:** `&`, `|` and `~` combine them, and `fx.where(condition, a,
  b)` chooses.
- **Functions:** `fx.mag`, `fx.dot`, `fx.cross`, `fx.grad`, `fx.div`,
  `fx.curl`, `fx.qcrit`, `fx.laplacian`, `fx.clamp`, `fx.points`, `fx.cells`,
  `fx.fsum` and the calculator's others.

`==`, `and`, `or`, `not` and chained comparisons are Python's own, so they
don't make expressions.

Every check of the text applies here too, since the expression is compiled
from the same calculator text. `expr.render(data)[0]` shows that text, with
the fields under their names in `data.fields`. Units are checked as well:
`np.sin(T)` of a temperature is refused.

A `td.Field` in an expression is that field, wherever the expression runs.
`fx.ref("velocity")` is a name instead. It is whatever
`data.fields["velocity"]` holds when the expression runs, such as a field a
solver replaces between steps, or each piece's own field over a
`td.PartitionedDataSet`.

## Take derivatives

`grad`, `div`, `curl`, `qcrit` and `laplacian` take a field or an expression:

- **Fields:** each field is differentiated through its cell's basis and
  geometry, as `td.gradients` differentiates it.
- **Expressions:** the kernel carries each value's derivative beside it and
  applies the chain rule. So `grad(p * p)` is exact wherever the expression is
  differentiable, for fields of any order, with nothing sampled or averaged.
- **Second derivatives** nest: `laplacian(p)`, `div(grad(u))`,
  `curl(curl(U))`.

<!-- tack-check: derivatives -->
```python
import numpy as np

import tack
import tack.dataset as td
import tack_expr as fx

tack.set_default_device("cpu")


@tack.func
def bowl(x):
    return x[0] * x[0] + x[1] * x[1]


data = td.uniform_grid((9, 9, 9), origin=(-1, -1, -1), spacing=(0.25, 0.25, 0.25))
data.fields["p"] = td.interpolate(data, bowl, td.H1(data, order=2))   # held exactly
p = data.fields["p"]

centres = fx.evaluate("coords", data).values.numpy()
grad_p2 = fx.evaluate("grad(p * p)", data)          # the chain rule, in the kernel
laplace = fx.evaluate("laplacian(p)", data)
same = (fx.grad(p * p)[0] + 1).eval(data)           # the same, without a string

r2 = centres[:, 0] ** 2 + centres[:, 1] ** 2
assert np.allclose(grad_p2.values.numpy()[:, 0], 4 * r2 * centres[:, 0], atol=1e-4)
assert np.allclose(laplace.values.numpy(), 4, atol=1e-3)
assert np.allclose(same.values.numpy(), grad_p2.values.numpy()[:, 0] + 1, atol=1e-5)
```

A field that carries a solver's own gradient
(`td.Field(..., gradient=g)`) gives `grad`, `div`, `curl` and `qcrit` that
gradient instead of its basis'.

## Choose where values are computed

`at` is `"cells"`, `"points"`, `"faces"`, or a space:
- **`L2` of any order:** the expression at every node of every cell, so an
  order-2 field's detail survives.
- **`H1`:** for an expression of continuous fields. With `project=True`, other
  expressions are evaluated into `L2` and averaged into `H1` by `td.project`.

A field with a basis (`H1`, `L2`, `Constant`) is evaluated through the basis
at each output location. Values stored on other entities need one of these
operators:

| Operator | Gives |
|---|---|
| `points(T)` | cell data averaged at each point, as `vtkCellDataToPointData` |
| `cells(T)` | point data averaged over each cell, as `vtkPointDataToCellData` |
| `owner(T)`, `neighbor(T)` | in a face expression, the cell value on side 0 or side 1 |
| `fsum(F)` | in a cell expression, the outward sum of a face field over the cell's faces |

Tensors and complex values are read as well:
- **Tensors.** A field stored as `tack.mat3`, as 9 components or as 6 symmetric
  ones (VTK's order: XX YY ZZ XY YZ XZ) is a 3x3 tensor. `grad` of a vector
  is its Jacobian, and `trace`, `det`, `inv`, `sym`, `dev`, `ddot`,
  `vonmises` and `eigvals` take tensors.
- **Complex values.** A field stored as two real fields, `u_real` and `u_imag`
  (or `_re`/`_im`), is the complex field `u`, as a harmonic analysis writes
  its response. `abs(u)` gives its amplitude, and `phase(u, pi / 4)` the
  response at 45 degrees.

## Work in units

A field's `unit` and the dataset's spatial unit carry through an expression:
- **Results.** A product's unit is the product of units. A derivative divides
  by the geometry's unit.
- **Conversion.** Units of one dimension are converted where they meet, in a
  sum, a comparison, `min`, `max` or a choice.
- **Refusal.** Units of different dimensions are refused where they meet.
- **Unknown units.** A field without a unit gives a result without one: the
  unit is never guessed.

<!-- tack-check: units -->
```python
import numpy as np

import tack
import tack.dataset as td
import tack_expr as fx

tack.set_default_device("cpu")

data = td.uniform_grid((5, 5, 5), spacing=(0.25, 0.25, 0.25), unit="m")
x = data.positions()
data.fields["p"] = td.Field(td.H1(data), tack.array(101325 + 1000 * x[:, 2]), unit="Pa")
data.fields["T"] = td.Field(td.H1(data), tack.array(300 + 10 * x[:, 0]), unit="K")
data.fields["rho_air"] = td.Field(td.H1(data), tack.array(np.full(len(x), 1.2, np.float32)))
data.fields["U"] = td.Field(td.H1(data), tack.array(np.tile([2.0, 0.0, 0.0], (len(x), 1)),
                                                    dtype=tack.vec3(tack.f32)), unit="m/s")

gauge = fx.evaluate("p - 101.325[kPa]", data, at="points")      # kPa met Pa: converted
assert gauge.unit == "Pa" and np.allclose(gauge.values.numpy(), 1000 * x[:, 2], atol=0.05)

per_mm = fx.evaluate("grad(T)", data, units="K/mm")             # 10 K/m is 0.01 K/mm
assert np.allclose(per_mm.values.numpy(), [0.01, 0, 0], atol=1e-6)

# rho_air is stored without a unit; the dictionary names it and gives it one.
names = fx.cgns().with_sources(Density="rho_air")
dynamic = fx.evaluate("0.5 * Density * mag2(U)", data, fields=names)
print(dynamic.unit)                                             # kg/(m*s**2), a pressure
assert np.allclose(dynamic.values.numpy(), 2.4)

try:
    fx.evaluate("p + U", data)
except fx.ExprError as error:                                   # a pressure plus a velocity
    print("refused:", str(error).splitlines()[0])
```

`101.325[kPa]` gives a number its own unit. A bare number in a sum or a
comparison takes the unit of what it meets, so `T > 300` compares kelvin with
kelvin. `units=` asks for the result in a unit of its dimension.

A `fx.FieldDictionary` maps the names an expression uses to what a dataset
stores, and gives units to fields stored without them:
- **Finding a field.** An entry finds its field by name, alias or a source
  you give.
- **Assembling a field.** Three component fields make a vector
  (`VelocityX`, `VelocityY`, `VelocityZ`). Six make a symmetric tensor, and a
  real and an imaginary part make a complex value.
- **`fx.cgns()`** names the CGNS SIDS flow quantities in SI.
- **`fields={"p": "kPa"}`** is the short form: a unit for each named field.

## Compute several at once

`fx.compile_many` compiles several expressions at one location into one
kernel. Each field read, derivative and conversion is then done once for all
of them. `fx.compile_expression` compiles one expression. Each compiled
object is called on the dataset as often as its fields change. Calling one
from `compile_many` gives a field per name:

<!-- tack-check: many-and-unstored -->
```python
import numpy as np

import tack
import tack.dataset as td
import tack_expr as fx

tack.set_default_device("cpu")

data = td.uniform_grid((17, 17, 17), origin=(-1, -1, -1), spacing=(0.125, 0.125, 0.125))
x = data.positions()
rotation = np.stack([-x[:, 1], x[:, 0], np.full(len(x), 0.5, np.float32)], axis=1)
data.fields["velocity"] = td.Field(td.H1(data), tack.array(rotation, dtype=tack.vec3(tack.f32)))

# Several expressions, one kernel: each field read and derivative taken once.
derived = fx.compile_many({"vorticity": "mag(curl(velocity))", "q": "qcrit(velocity)",
                           "helicity": "dot(velocity, curl(velocity))"}, data)
out = derived(data)                              # {name: td.Field}; call again as data changes
assert np.allclose(out["q"].values.numpy(), 1, atol=1e-4)
assert np.allclose(out["helicity"].values.numpy(), 1, atol=1e-4)

# Not stored: computed wherever a filter reads it.
shell = td.contour(data, fx.field("mag(velocity)"), 1.0)
r = np.linalg.norm(shell.positions()[:, :2], axis=1)
assert shell.num_cells > 0 and np.allclose(r, np.sqrt(0.75), atol=0.02)

# Stored under a name: other expressions read it, and it follows velocity.
data.fields["speed"] = fx.field("mag(velocity)", data)
fast = fx.evaluate("speed > 1 ? speed : 0", data, at="points")
assert np.allclose(fast.values.numpy().max(), np.sqrt(2 + 0.25), atol=1e-5)
```

## Use an expression without storing it

`fx.field(text)` is a field that computes the expression wherever an algorithm
reads it. Nothing is stored. The second part of the example contours the
speed without making a speed array:
- **Crossings.** Each crossing is the expression's own root along its edge.
- **Order.** The contour subdivides by the largest order of the fields the
  expression reads.
- **Refinement.** Each cell's values are bounded by interval arithmetic, so a
  subdivided contour refines only the cells that can hold the level.

Given a dataset, `fx.field(text, data)` can also be stored under a name, as
the third part does:
- **In other expressions.** They read it as a field, and it expands into
  their kernel, so it always follows the fields it reads.
- **Through filters.** A filter makes it again on its output, over the fields
  carried there.
- **Written out.** `dataset_to_vtk` writes its values.

`fx.function(text, data)` gives the expression as a
[device class](../core/device-classes.md) with `value(cells, c, pc)`, for a
kernel of your own: a threshold or a reduction that reads it without storing
it. `fx.for_each` runs such a kernel.

## Over pieces

`fx.evaluate` of a `td.PartitionedDataSet` runs piece by piece and returns a
field for each piece. Over pieces
with a ghost layer, what each piece owns is the whole mesh's answer. Text
names each piece's own fields. An expression built with Python operators names
them with `fx.ref("name")`, since a `td.Field` belongs to one piece.

An expression that reads neighbours' values, such as `points()` or
`cells()`, needs current ghost values. On pieces made by `td.ghost_cells` or
`td.ghost_points`, `update_ghosts=True` first refreshes the ghost values of
the fields the expression reads, from the pieces that own them, across MPI
ranks too. See [filters over pieces](over-pieces.md).

## In a VTK pipeline

`tack.interop.vtk.algorithms.Calculator` is the expression as a VTK
algorithm, beside the [other wrapped filters](vtk.md#put-tacks-filters-in-a-vtk-pipeline).
It takes `vtkArrayCalculator`'s syntax, and fields of any order and curved
cells are evaluated through each cell's own map:

<!-- tack-check: calculator -->
```python
import numpy as np
import vtkmodules.vtkCommonDataModel  # registers VTK's pythonic constructors
from vtkmodules.vtkFiltersCore import vtkArrayCalculator
from vtkmodules.vtkImagingCore import vtkRTAnalyticSource

import tack
from tack.interop.vtk.algorithms import Calculator

tack.set_default_device("cpu")

source = vtkRTAnalyticSource(whole_extent=(-10, 10, -10, 10, -10, 10))
calculator = Calculator(expression="sqrt(RTData) * 2", at="points", name="result")
ours = (source >> calculator)()
assert calculator.error is None

# The same by VTK.
theirs = vtkArrayCalculator(function="sqrt(RTData) * 2", result_array_name="result")
theirs.AddScalarArrayName("RTData")
reference = (source >> theirs)()
assert np.allclose(np.asarray(ours.point_data["result"]),
                   np.asarray(reference.point_data["result"]), rtol=1e-5)
```

## More

tack-expr's own [usage guide](https://github.com/Kitware/tack/blob/main/packages/tack-expr/GUIDE.md)
lists every function and operator, and covers tensors, complex values and the
field dictionary in full. Its
[README](https://github.com/Kitware/tack/blob/main/packages/tack-expr/README.md)
has the design, the checks against VTK and NumPy, and the measurements. Its
examples, in `packages/tack-expr/examples/`, check their answers against
fields whose values are known.
