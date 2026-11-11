# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Expressions as device functions in other kernels, compiled expressions reused after the
data changes, and the module cache."""

import os
import time

import numpy as np
import pytest
import tack_expr as fx
from _expr_checks import assert_close, values, vector_array, with_fields

import tack
import tack.dataset as td


@pytest.fixture
def pointdata(backend):
    return with_fields(td.uniform_grid((12, 9, 7), spacing=(0.1, 0.2, 0.05)), np.random.default_rng(2), td.H1)


@pytest.fixture
def celldata(backend):
    return with_fields(td.uniform_grid((12, 9, 7), spacing=(0.1, 0.2, 0.05)), np.random.default_rng(2), td.Constant)


@tack.kernel
def above(cells, q, out, level):
    for c in cells:
        out[cells.entity_id(c)] = 1.0 if q.value(cells, c, cells.parametric_center()) > level else 0.0


@tack.kernel
def face_values(faces, q, out):
    for f in faces:
        out[faces.entity_id(f)] = q.value(faces, f)


def test_a_cell_expression_inside_a_threshold_kernel(pointdata):
    expr = "mag(V) * T + cells(P)"
    mask = tack.zeros((pointdata.num_cells,))
    fx.for_each(above, pointdata, fx.function(expr, pointdata), mask, 0.6)
    assert_close(mask.numpy(), (values(expr, pointdata) > 0.6).astype(float))


def test_a_face_expression_inside_a_face_kernel(celldata):
    expr = "neighbor(T) - owner(T) * 2"
    out = tack.zeros((celldata.topology.faces().num_faces,))
    fx.for_each(face_values, celldata, fx.function(expr, celldata, at="faces"), out)
    assert_close(out.numpy(), values(expr, celldata, at="faces"))


@tack.func
def vfun(x):
    return tack.vec3(x[0] * x[1], x[1] * x[1] + x[2], x[0] - 2.0 * x[2] * x[1])


def test_reuse_after_a_field_is_replaced(backend):
    data = td.uniform_grid((6, 5, 4), origin=(0.2, -0.5, 1.0), spacing=(0.3, 0.25, 0.4))
    data.fields["U"] = td.interpolate(data, vfun, td.H1(data, order=2), dtype=tack.vec3())
    div = fx.compile_expression("div(U)", data)
    div(data)
    rng = np.random.default_rng(4)
    space = data.fields["U"].space
    data.fields["U"] = td.Field(space, vector_array(rng.random((space.size, 3), dtype=np.float32) - 0.5))
    fq = td.flow_quantities(data, td.gradients(data, "U", at="cells"))
    assert_close(div(data).values.numpy(), fq.divergence.values.numpy())


def test_reuse_after_values_change_in_place(celldata):
    points = fx.compile_expression("points(T) + 1", celldata, at="points")
    points(celldata)
    tack.copyto(celldata.fields["T"].values, np.random.default_rng(5).random(celldata.num_cells, dtype=np.float32))
    assert_close(points(celldata).values.numpy(), td.project(celldata, "T", td.H1(celldata)).values.numpy() + 1)


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(fx.parser, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(fx.parser, "_compiled", {})
    return tmp_path


def test_clear_cache_then_recompile(pointdata, cache_dir):
    first = values("T + P * 3", pointdata, at="points")
    assert any(n.endswith(".py") for n in os.listdir(cache_dir))
    assert fx.clear_cache() >= 1
    assert_close(values("T + P * 3", pointdata, at="points"), first)


def test_clear_cache_keeps_modules_used_recently(pointdata, cache_dir):
    stale = cache_dir / "fx_0000000000000000.py"
    stale.write_text("# a stale module\n")
    past = time.time() - 40 * 86400
    os.utime(stale, (past, past))
    values("T * 3 + 0.125", pointdata, at="points")
    fresh = {n for n in os.listdir(cache_dir) if n.endswith(".py")} - {stale.name}
    assert fx.clear_cache(older_than_days=30) == 1
    assert not stale.exists()
    assert fresh <= set(os.listdir(cache_dir))
