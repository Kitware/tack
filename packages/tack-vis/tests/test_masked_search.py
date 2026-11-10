# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Hidden and refined cells left out by the cell search itself (issue #402).

``probe``, ``advect``, ``streamlines`` and their pieces' paths used to extract
the cells that are not hidden or refined (``ghosts.visible``) into an
unstructured copy, and build that copy a cell locator, on every call: every
AMR box below the finest level paid it. Now the search skips those cells
(``CellLocator.find``'s ``hidden``, ``by_id._MeshSearch.with_masks``), so the
dataset keeps its topology, its cached locator and its structured search. The
answers must be the extracted dataset's, bit for bit: each case here runs the
algorithm on the dataset and on ``visible(dataset)`` and compares.
"""

import numpy as np
import pytest

import tack
import tack.dataset as td
from tack.dataset.ghosts import visible
from tack.dataset.locator import cached_locator

pytestmark = pytest.mark.quick      # the inner-loop set; see CLAUDE.md, Tests


def _bits(a):
    return np.asarray(a).view(np.uint8)


def _masked(data, fraction=0.15, seed=2):
    rng = np.random.default_rng(seed)
    marks = np.zeros(data.num_cells, np.uint8)
    marks[rng.random(data.num_cells) < fraction] = td.Ghosts.HIDDENCELL
    marks[rng.random(data.num_cells) < fraction] |= td.Ghosts.REFINEDCELL
    marks[rng.random(data.num_cells) < 0.1] |= td.Ghosts.DUPLICATECELL
    data.ghosts = td.Ghosts(cells=tack.array(marks, tack.u8))
    return data


def _fields(data):
    x = data.geometry.values.numpy().astype(float)
    dtype = data.dtype
    data.fields["s"] = td.Field(td.H1(data), tack.array(
        (np.sin(x[:, 0]) + x[:, 1] ** 2).astype(dtype.numpy_dtype)))
    data.fields["v"] = td.Field(td.H1(data), tack.array(
        np.c_[-(x[:, 1] - 2.0), x[:, 0] - 2.0, 0.1 + 0.0 * x[:, 2]].astype(dtype.numpy_dtype),
        tack.vec(3, dtype)))
    data.fields["c"] = td.Field(td.Constant(data), tack.array(
        np.arange(data.num_cells).astype(dtype.numpy_dtype)))
    return data


def _dtypes():
    from tack.runtime.dispatch import get_backend

    return [tack.f32] + ([tack.f64] if get_backend().supports_f64 else [])


def _meshes(dtype):
    rng = np.random.default_rng(4)
    return {
        "image": td.uniform_grid((12, 10, 8), spacing=(0.4, 0.4, 0.4), dtype=dtype),
        "rectilinear": td.rectilinear_grid(*[np.cumsum(rng.uniform(0.2, 0.6, n))
                                             for n in (12, 10, 8)], dtype=dtype),
        "tetrahedra": td.tetrahedralize(td.uniform_grid((10, 8, 6), spacing=(0.5, 0.5, 0.5),
                                                        dtype=dtype)),
        "polyhedra": td.as_polyhedra(td.uniform_grid((8, 7, 6), spacing=(0.6, 0.6, 0.6),
                                                     dtype=dtype)),
    }


@pytest.mark.parametrize("kind", ["image", "rectilinear", "tetrahedra", "polyhedra"])
def test_probe_leaves_masked_cells_out_as_extracting_them_did(backend, kind):
    for dtype in _dtypes():
        data = _fields(_masked(_meshes(dtype)[kind]))
        x = data.geometry.values.numpy().astype(float)
        lo, hi = x.min(axis=0), x.max(axis=0)
        points = np.random.default_rng(5).uniform(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo),
                                                  (3000, 3)).astype(dtype.numpy_dtype)
        a = td.probe(data, points)
        b = td.probe(visible(data), points)
        for name in ("s", "v", "c", "valid"):
            assert np.array_equal(_bits(a.fields[name].values.numpy()),
                                  _bits(b.fields[name].values.numpy())), name
        valid = a.fields["valid"].values.numpy()
        assert 0 < valid.sum() < len(points)


@pytest.mark.parametrize("kind", ["image", "tetrahedra"])
def test_advection_leaves_masked_cells_out_as_extracting_them_did(backend, kind):
    for dtype in _dtypes():
        data = _fields(_masked(_meshes(dtype)[kind], fraction=0.03))
        seeds = np.array([[2.5, 1.5, 0.5], [3.0, 2.2, 1.0], [1.2, 1.0, 1.5]],
                         dtype.numpy_dtype)
        for neighbors in (False, True):
            a = td.streamlines(data, "v", seeds, 0.05, 150, neighbors=neighbors)
            b = td.streamlines(visible(data), "v", seeds, 0.05, 150, neighbors=neighbors)
            assert a.num_points == b.num_points and a.num_points > 20
            assert np.array_equal(_bits(a.geometry.values.numpy()),
                                  _bits(b.geometry.values.numpy()))
        a = td.advect(data, "v", seeds, 0.05, 150)
        b = td.advect(visible(data), "v", seeds, 0.05, 150)
        assert np.array_equal(_bits(a.geometry.values.numpy()), _bits(b.geometry.values.numpy()))
        assert np.array_equal(a.fields["status"].values.numpy(),
                              b.fields["status"].values.numpy())


def test_pieces_leave_masked_cells_out_as_extracting_them_did(backend):
    pieces = td.split(td.uniform_grid((16, 12, 10), spacing=(0.4, 0.4, 0.4)), (2, 2, 1))
    for seed, piece in enumerate(pieces):
        _fields(_masked(piece, seed=seed))
    points = np.random.default_rng(6).uniform(0.0, 6.0, (3000, 3)).astype(np.float32)
    a = td.probe(pieces, points)
    b = td.probe(td.PartitionedDataSet([visible(p) for p in pieces]), points)
    for name in ("s", "valid"):
        assert np.array_equal(_bits(a.fields[name].values.numpy()),
                              _bits(b.fields[name].values.numpy()))


def test_a_first_small_query_from_the_cells_leaves_them_out_too(backend):
    data = _fields(_masked(_meshes(tack.f32)["tetrahedra"]))
    few = np.random.default_rng(7).uniform(0.0, 4.0, (5, 3)).astype(np.float32)
    a = td.CellLocator(data).find(few, hidden=data.ghosts.cells)
    b = td.CellLocator(visible(data)).find(few)
    assert np.array_equal(a.cells.numpy() >= 0, b.cells.numpy() >= 0)
    assert np.array_equal(_bits(a.pcs.numpy()), _bits(b.pcs.numpy()))


def test_the_dataset_keeps_its_topology_and_locator_across_calls(backend):
    # Extracting made a new topology, and so a new locator, on every call; a
    # locator given was refused. Now one is kept, and one given is taken.
    data = _fields(_masked(td.uniform_grid((10, 9, 8))))
    points = np.random.default_rng(8).uniform(0.0, 7.0, (500, 3)).astype(np.float32)
    first = td.probe(data, points)
    kept = data.topology._locator
    assert kept is not None and kept.structured is not None     # the arithmetic search
    second = td.probe(data, points, locator=cached_locator(data))
    assert data.topology._locator is kept
    assert np.array_equal(first.fields["s"].values.numpy(), second.fields["s"].values.numpy())
    seeds = np.array([[3.0, 3.0, 3.0]], np.float32)
    td.streamlines(data, "v", seeds, 0.05, 20, locator=kept)
    # The marks are the query's: the same locator serves the dataset without them.
    plain = td.DataSet(data.topology, data.geometry, fields=dict(data.fields))
    whole = td.probe(plain, points)
    assert whole.fields["valid"].values.numpy().sum() >= first.fields["valid"].values.numpy().sum()
    assert data.topology._locator is kept
