# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Expressions on pieces with a ghost layer: what each piece owns is the whole's, and chained
evaluations refresh the ghosts between them when asked."""

import pytest
import tack_expr as fx
from _expr_checks import assert_close, with_position_flux

import tack.dataset as td

pytestmark = pytest.mark.quick


@pytest.mark.parametrize(("expr", "at"), [
    ("RTData * 2 + 1", "points"), ("cells(RTData)", "cells"), ("mag(grad(RTData))", "cells"),
    ("points(C) - RTData", "points"), ("fsum(flux) * C", "cells"),
])
def test_owned_values_are_the_wholes(backend, expr, at):
    whole = td.wavelet(10)
    pieces = td.split(whole, 4, ghost_layers=1)
    for data in [whole, *pieces]:
        with_position_flux(data)
        fx.evaluate("cells(RTData)", data, at="cells", name="C")
    fx.evaluate(expr, whole, at=at, name="r")
    fx.evaluate(expr, pieces, at=at, name="r")
    a, b = td.statistics(whole, "r"), td.statistics(pieces, "r")
    assert_close([b.mean, b.min, b.max, b.n], [a.mean, a.min, a.max, a.n], tol=1e-5)


CHAIN = (("cells(RTData)", "cells", "c"), ("points(c)", "points", "p"),
         ("mag(grad(p))", "cells", "g"), ("points(g)", "points", "q"))


def chained(data, refresh):
    for expr, at, name in CHAIN:
        fx.evaluate(expr, data, at=at, name=name, update_ghosts=refresh)
    return td.statistics(data, "q")


def chain_error(refresh):
    ref = chained(td.wavelet(10), False)
    got = chained(td.ghost_cells(td.split(td.wavelet(10), 4)), refresh)
    return max(abs(got.mean - ref.mean) / abs(ref.mean), abs(got.max - ref.max) / abs(ref.max))


def test_update_ghosts_keeps_a_chain_of_evaluations_the_wholes(backend):
    assert chain_error(True) < 1e-5


def test_without_the_refresh_the_ghost_rim_reaches_the_owned_values(backend):
    assert chain_error(False) > 1e-4
