# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""50 -- Field expressions: VTK calculator syntax over tack.dataset fields.

A velocity field whose answers are known exactly -- a rigid rotation about the z axis with an
axial drift, v = (-y, x, w) -- and a pressure p = x^2 + y^2, both held exactly by order-2 H1
fields, so every expression's result is checked:

- at points: |v| = sqrt(x^2 + y^2 + w^2), and a conditional on p;
- derivatives at cells: div(v) = 0, curl(v) = (0, 0, 2), qcrit(v) = 1, grad(p) = (2x, 2y, 0)
  and laplacian(p) = 4;
- several expressions compiled into one kernel;
- a compiled expression called again after the velocity changes in place.

Each expression compiles to one Tack kernel the first time it is seen; later calls, with any
constants, reuse it.

Usage:
  uv run python packages/tack-expr/examples/50_field_expressions.py
  uv run python packages/tack-expr/examples/50_field_expressions.py --arch cuda --size 33
"""

import argparse

import numpy as np
import tack_expr as fx

import tack
import tack.dataset as td

_parser = argparse.ArgumentParser()
_parser.add_argument('--arch', default='cpu',
                     choices=['cpu', 'metal', 'cuda', 'hip', 'level_zero'])
_parser.add_argument('--size', type=int, default=17, help='points per axis (default 17)')
_args = _parser.parse_args()
tack.set_default_device(_args.arch)

W = 0.2                                     # the axial drift, as in rotation() below


@tack.func
def rotation(x):
    return tack.vec3(-x[1], x[0], 0.2)


@tack.func
def bowl(x):
    return x[0] * x[0] + x[1] * x[1]


n = _args.size
data = td.uniform_grid((n, n, n), origin=(-1.0, -1.0, -1.0), spacing=(2.0 / (n - 1),) * 3)
data.fields["velocity"] = td.interpolate(data, rotation, td.H1(data, order=2), dtype=tack.vec3())
data.fields["p"] = td.interpolate(data, bowl, td.H1(data, order=2))
print(f"{n}^3 points, {data.num_cells} cells; v = (-y, x, {W}), p = x^2 + y^2")


def check(label, got, ref, tol=1e-4):
    got = np.asarray(got, np.float64)
    err = float(np.max(np.abs(got - np.broadcast_to(np.asarray(ref, np.float64), got.shape))))
    print(f"  {label:48} max error {err:.1e}")
    assert err < tol, label


def at(expr, where="cells"):
    return fx.evaluate(expr, data, at=where).values.numpy()


print("at points:")
x = at("coords", "points")
speed = np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2 + W * W)
check("mag(velocity)", at("mag(velocity)", "points"), speed)
check("p < 0.25 ? mag(velocity) : 0", at("p < 0.25 ? mag(velocity) : 0", "points"),
      np.where(x[:, 0] ** 2 + x[:, 1] ** 2 < 0.25, speed, 0.0))

print("derivatives at cells:")
xc = at("coords")
check("div(velocity) = 0", at("div(velocity)"), 0.0)
check("curl(velocity) = (0, 0, 2)", at("curl(velocity)"), [0.0, 0.0, 2.0])
check("qcrit(velocity) = 1", at("qcrit(velocity)"), 1.0)
check("grad(p) = (2x, 2y, 0)", at("grad(p)"), np.stack([2 * xc[:, 0], 2 * xc[:, 1], 0 * xc[:, 2]], axis=1))
check("laplacian(p) = 4", at("laplacian(p)"), 4.0, tol=1e-3)

print("one kernel for several expressions:")
derived = fx.compile_many({"vorticity": "mag(curl(velocity))", "q": "qcrit(velocity)",
                           "helicity": "dot(velocity, curl(velocity))"}, data)(data)
check("vorticity = 2", derived["vorticity"].values.numpy(), 2.0)
check("q = 1", derived["q"].values.numpy(), 1.0)
check(f"helicity = 2 w = {2 * W:g}", derived["helicity"].values.numpy(), 2 * W)

print("a compiled expression after the velocity changes in place:")
omega_z = fx.compile_expression("curl(velocity)[2]", data)
check("curl(velocity)[2] = 2", omega_z(data).values.numpy(), 2.0)
velocity = data.fields["velocity"].values
tack.copyto(velocity, velocity.numpy() * 3)
check("after v *= 3: curl(velocity)[2] = 6", omega_z(data).values.numpy(), 6.0, tol=3e-4)

print("\nthe kernel for mag(velocity) * p at points:")
print(fx.compile_expression("mag(velocity) * p", data, at="points").source)
