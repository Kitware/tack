# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""compile_many: several expressions in one kernel give what one kernel each gives."""

import numpy as np
import pytest
import tack_expr as fx
from _expr_checks import vector_array

import tack
import tack.dataset as td

pytestmark = pytest.mark.quick

SETS = {
    "points": {"cp": "2 * pressure / 0.0064", "ke": "0.5 * mag2(velocity)", "u2": "velocity * pressure"},
    "cells": {"vorticity": "mag(curl(velocity))", "q": "qcrit(velocity)",
              "helicity": "dot(velocity, curl(velocity))", "dp": "grad(pressure * speed)",
              "pc": "cells(pressure) + 1"},
    "L2 order 1": {"vorticity": "mag(curl(velocity))", "w": "velocity * 2 - coords"},
}


@pytest.fixture
def data(backend):
    rng = np.random.default_rng(3)
    d = td.uniform_grid((20, 16, 12), spacing=(0.1, 0.1, 0.1))
    n = d.num_points
    d.fields["velocity"] = td.Field(td.H1(d), vector_array(rng.random((n, 3), dtype=np.float32) - 0.5))
    d.fields["pressure"] = td.Field(td.H1(d), tack.array(rng.random(n, dtype=np.float32)))
    d.fields["speed"] = td.magnitude(d, "velocity")
    return d


@pytest.mark.parametrize("where", list(SETS))
def test_one_kernel_matches_one_kernel_each(data, where):
    at = td.L2(data, order=1) if where.startswith("L2") else where
    exprs = SETS[where]
    fields = fx.compile_many(exprs, data, at=at)(data)
    for name, text in exprs.items():
        got, ref = fields[name].values.numpy(), fx.evaluate(text, data, at=at).values.numpy()
        np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-6, err_msg=f"{name} = {text}")


def test_world_gradients_are_shared(data):
    source = fx.compile_many(SETS["cells"], data).source
    assert source.count("_world_gradient(dom, ") == 5


def test_an_output_passed_in_is_filled_in_place(data):
    out = {"cp": tack.empty((data.num_points,))}
    filled = fx.compile_many(SETS["points"], data, at="points")(data, out=out)
    assert filled["cp"].values is out["cp"]
