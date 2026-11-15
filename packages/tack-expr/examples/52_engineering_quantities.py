# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Engineering quantities in expressions: units, a field dictionary, tensors and complex values.

A harmonic response of a plate stored the way a solver writes it, in mm and MPa: a complex
displacement as two real arrays and a complex stress as twelve components. The expressions
name quantities, not arrays; units are checked and converted; the stress is a tensor.
"""

import numpy as np
import tack_expr as fx

import tack
import tack.dataset as td

tack.set_default_device("cpu")

data = td.uniform_grid((21, 11, 3), spacing=(5.0, 5.0, 2.0), unit="mm")
cells = td.Constant(data)
x = fx.evaluate("coords", data).values.numpy()
n = data.num_cells
rng = np.random.default_rng(0)

def stored(values, dtype=None):
    out = tack.empty((n,), dtype or data.dtype)
    tack.copyto(out, values)
    return out


wave = np.exp(1j * np.pi * x[:, 0] / 100.0)
for part, f in (("real", np.real), ("imag", np.imag)):
    disp = np.stack([0.01 * f(wave), 0.002 * f(1j * wave), 0 * x[:, 2]], axis=1)
    data.fields[f"disp_{part}"] = td.Field(cells, stored(disp, tack.vec3(data.dtype)))
    for c in ("XX", "YY", "ZZ", "XY", "YZ", "XZ"):
        scale = 80.0 if c in ("XX", "YY") else 15.0
        data.fields[f"S{c}_{part}"] = td.Field(cells, stored(f(scale * wave * (1 + 0.1 * rng.random(n)))))

names = (fx.FieldDictionary(length="mm")
         .define("Displacement", "mm", aliases=("u",), complex=("disp_real", "disp_imag")))
for c in ("XX", "YY", "ZZ", "XY", "YZ", "XZ"):
    names.define(f"Stress{c}", "MPa", complex=(f"S{c}_real", f"S{c}_imag"))
names.define("Stress", "MPa", aliases=("sigma",),
             components=tuple(f"Stress{c}" for c in ("XX", "YY", "ZZ", "XY", "YZ", "XZ")))

amplitude = fx.evaluate("mag(u)", data, fields=names, units="um")
assert np.allclose(amplitude.values.numpy(), 1e3 * np.hypot(0.01, 0.002), rtol=1e-4)
print("displacement amplitude", amplitude.values.numpy().max(), amplitude.unit)

for degrees in (0, 45, 90):
    vm = fx.evaluate(f"vonmises(phase(sigma, {degrees}[deg]))", data, fields=names, units="MPa")
    print(f"von Mises at {degrees:2d} degrees, MPa: max {vm.values.numpy().max():.1f}")

principal = fx.evaluate("eigvals(real(sigma))[0] - 50[MPa]", data, fields=names)
parts = {c: data.fields[f"S{c}_real"].values.numpy() for c in ("XX", "YY", "ZZ", "XY", "YZ", "XZ")}
tensors = np.stack([[parts["XX"], parts["XY"], parts["XZ"]], [parts["XY"], parts["YY"], parts["YZ"]],
                    [parts["XZ"], parts["YZ"], parts["ZZ"]]]).transpose(2, 0, 1)
assert np.allclose(principal.values.numpy(), np.linalg.eigvalsh(tensors)[:, 2] - 50.0, atol=1e-3)
print("largest principal stress less 50 MPa:", principal.values.numpy().max(), principal.unit)

try:
    fx.evaluate("real(sigma_XX) + real(u)[0]", data, fields=names)
except fx.ExprError as error:
    print("refused:", error.args[0].splitlines()[0])
else:
    raise AssertionError("mm plus MPa was not refused")
