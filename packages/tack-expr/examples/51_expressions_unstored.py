# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""51 -- Field expressions without storing them: in a kernel of your own, in a contour, on pieces.

The rotation of example 50, v = (-y, x, w), whose speed |v| = s on the cylinder of radius
sqrt(s^2 - w^2), so each result is checked:

- fx.function: an expression as a device function that a threshold kernel calls at each cell
  center, against the speed computed on the host;
- td.contour of the text: the surface |v| = s, the expression evaluated where the contour reads it and
  each crossing its own root, on cells subdivided by the velocity's order (2), against the
  cylinder's radius; nothing is added to the fields;
- pieces with a ghost layer: a chain of evaluations that average over neighbours gives what
  the whole mesh gives when update_ghosts=True refreshes the ghosts between them;
- the time one kernel for four derived fields takes against one kernel each.

Usage:
  uv run python packages/tack-expr/examples/51_expressions_unstored.py
  uv run python packages/tack-expr/examples/51_expressions_unstored.py --arch cuda --size 65
"""

import argparse
import time

import numpy as np
import tack_expr as fx

import tack
import tack.dataset as td

_parser = argparse.ArgumentParser()
_parser.add_argument('--arch', default='cpu',
                     choices=['cpu', 'metal', 'cuda', 'hip', 'level_zero'])
_parser.add_argument('--size', type=int, default=33, help='points per axis (default 33)')
_args = _parser.parse_args()
tack.set_default_device(_args.arch)

W = 0.2                                     # the axial drift, as in rotation() below
S = 0.7                                     # the speed of the threshold and the contour


@tack.func
def rotation(x):
    return tack.vec3(-x[1], x[0], 0.2)


n = _args.size
data = td.uniform_grid((n, n, n), origin=(-1.0, -1.0, -1.0), spacing=(2.0 / (n - 1),) * 3)
data.fields["velocity"] = td.interpolate(data, rotation, td.H1(data, order=2), dtype=tack.vec3())
print(f"{n}^3 points; v = (-y, x, {W}); |v| = {S} on the cylinder of radius {np.sqrt(S * S - W * W):.4f}")


@tack.kernel
def faster_than(cells, speed, out, level):
    for c in cells:
        out[cells.entity_id(c)] = 1.0 if speed.value(cells, c, cells.parametric_center()) > level else 0.0


mask = tack.zeros((data.num_cells,))
fx.for_each(faster_than, data, fx.function("mag(velocity)", data), mask, S)
xc = fx.evaluate("coords", data).values.numpy().astype(np.float64)
expected = np.sqrt(xc[:, 0] ** 2 + xc[:, 1] ** 2 + W * W) > S
assert np.array_equal(mask.numpy() > 0.5, expected)
print(f"threshold kernel calling mag(velocity): {int(expected.sum())} of {data.num_cells} cells faster than {S}")

surface = td.contour(data, fx.field("mag(velocity)"), S)
xs = np.asarray(surface.positions(), np.float64)
err = float(np.abs(np.hypot(xs[:, 0], xs[:, 1]) - np.sqrt(S * S - W * W)).max())
print(f"contour of mag(velocity) = {S}: {surface.num_cells} triangles, radius error {err:.1e}")
assert err < 1e-4
assert set(data.fields) == {"velocity"}

chain = (("cells(RTData)", "cells", "c"), ("points(c)", "points", "p"),
         ("mag(grad(p))", "cells", "g"), ("points(g)", "points", "q"))


def chained(d, refresh=False):
    for expr, where, name in chain:
        fx.evaluate(expr, d, at=where, name=name, update_ghosts=refresh)
    return td.statistics(d, "q")


whole = chained(td.wavelet(10))
print("a chain of four evaluations on 4 pieces with a ghost layer, owned mean against the whole's:")
for refresh in (False, True):
    pieces = chained(td.ghost_cells(td.split(td.wavelet(10), 4)), refresh)
    err = abs(pieces.mean - whole.mean) / abs(whole.mean)
    print(f"  update_ghosts={refresh!s:5}  relative error {err:.1e}")
assert err < 1e-5

four = {"vorticity": "mag(curl(velocity))", "q_criterion": "qcrit(velocity)",
        "helicity": "dot(velocity, curl(velocity))", "omega_z": "curl(velocity)[2]"}
one_each = [fx.compile_expression(t, data) for t in four.values()]
fused = fx.compile_many(four, data)


def timed(fn, reps=10):
    fn()
    tack.sync()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    tack.sync()
    return (time.perf_counter() - t0) / reps * 1e3


print(f"four derived fields: one kernel each {timed(lambda: [c(data) for c in one_each]):.2f} ms, "
      f"one kernel for all four {timed(lambda: fused(data)):.2f} ms")
