# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Hidden and refined cells left out in place by ``tetrahedralize`` (#403) and
``integrate`` (#404), rather than by extracting the other cells first
(``ghosts.visible``).

Extracting made a structured grid's hexahedra unstructured, which
``tetrahedralize`` splits in 6 by cones, not in 5 by parity, so every AMR box
below the finest level split unlike the finest ones; ``integrate`` extracted
cells that its ``owned_values`` already leaves out.
"""

import numpy as np
import pytest

import tack
import tack.dataset as td
from tack.dataset.ghosts import visible

pytestmark = pytest.mark.quick      # the inner-loop set; see CLAUDE.md, Tests

_MASKED = td.Ghosts.HIDDENCELL | td.Ghosts.REFINEDCELL


def _masked(data, seed=3):
    rng = np.random.default_rng(seed)
    marks = np.zeros(data.num_cells, np.uint8)
    marks[rng.random(data.num_cells) < 0.2] = td.Ghosts.HIDDENCELL
    marks[rng.random(data.num_cells) < 0.2] |= td.Ghosts.REFINEDCELL
    marks[rng.random(data.num_cells) < 0.1] |= td.Ghosts.DUPLICATECELL
    data.ghosts = td.Ghosts(cells=tack.array(marks, tack.u8))
    return data, marks


def _dtypes():
    from tack.runtime.dispatch import get_backend

    return [tack.f32] + ([tack.f64] if get_backend().supports_f64 else [])


def test_a_grid_with_refined_cells_splits_its_other_hexahedra_in_five(backend):
    plain = td.tetrahedralize(td.uniform_grid((7, 6, 5)))
    data, marks = _masked(td.uniform_grid((7, 6, 5)))
    data.fields["c"] = td.Field(td.Constant(data), tack.array(
        np.arange(data.num_cells, dtype=np.float32)))
    out = td.tetrahedralize(data)
    kept = (marks & _MASKED) == 0
    # Each kept hexahedron's 5 tetrahedra, as the grid without marks splits it.
    expected = plain.topology.connectivity.numpy().reshape(-1, 5, 4)[kept].reshape(-1, 4)
    assert np.array_equal(out.topology.connectivity.numpy().reshape(-1, 4), expected)
    assert out.num_points == data.num_points               # the points are the same
    assert np.array_equal(out.fields["c"].values.numpy(),
                          np.repeat(np.flatnonzero(kept), 5).astype(np.float32))
    assert np.array_equal(out.ghosts.cells.numpy(), np.repeat(marks[kept], 5))


@pytest.mark.parametrize("make", [
    lambda: td.tetrahedralize(td.uniform_grid((6, 5, 4))),
    lambda: td.extract_cells(td.uniform_grid((6, 5, 4)), np.arange(60)),
])
def test_other_cells_split_as_the_extracted_ones_did(backend, make):
    data, _ = _masked(make())
    a = td.tetrahedralize(data)
    b = td.tetrahedralize(visible(data))
    corners_a = a.geometry.values.numpy()[a.topology.connectivity.numpy().reshape(-1, 4)]
    corners_b = b.geometry.values.numpy()[b.topology.connectivity.numpy().reshape(-1, 4)]
    assert a.num_cells == b.num_cells and np.array_equal(corners_a, corners_b)
    assert a.num_points == data.num_points


@pytest.mark.parametrize("make", [
    lambda dtype: td.uniform_grid((9, 8, 7), spacing=(0.3, 0.4, 0.5), dtype=dtype),
    lambda dtype: td.tetrahedralize(td.uniform_grid((7, 6, 5), dtype=dtype)),
    lambda dtype: td.as_polyhedra(td.uniform_grid((6, 5, 4), dtype=dtype)),
])
def test_integrate_leaves_masked_cells_out_as_extracting_them_did(backend, make):
    rng = np.random.default_rng(4)
    for dtype in _dtypes():
        data, _ = _masked(make(dtype))
        x = data.geometry.values.numpy().astype(float)
        data.fields["s"] = td.Field(td.H1(data), tack.array(
            (np.sin(x[:, 0]) + x[:, 1] * x[:, 2]).astype(dtype.numpy_dtype)))
        data.fields["v"] = td.Field(td.H1(data), tack.array(x.astype(dtype.numpy_dtype),
                                                           tack.vec(3, dtype)))
        data.fields["c"] = td.Field(td.Constant(data), tack.array(
            rng.random(data.num_cells).astype(dtype.numpy_dtype)))
        a = td.integrate(data)
        b = td.integrate(visible(data))
        assert a.measure == b.measure and set(a.values) == set(b.values)
        for name in b.values:
            assert np.array_equal(np.asarray(a.values[name]), np.asarray(b.values[name])), name
