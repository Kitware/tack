# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Pieces combined only when they agree (#411): ``probe``, ``integrate``,
``merge_pieces`` and ``ghost_cells`` over a ``PartitionedDataSet`` compare their
pieces' fields -- names, element types, units -- and refuse a mismatch before
anything is exchanged, on every rank (``_empty_ranks_mpi.py``'s ``refused/``
cases). They took the first piece's names and units as everyone's: a probe
failed after its points were sent (and over MPI hung), and values in metres
and centimetres were summed, or merged, labelled metres.
"""

import math

import numpy as np
import pytest

import tack
import tack.dataset as td

pytestmark = pytest.mark.quick      # the inner-loop set; see CLAUDE.md, Tests

_POINTS = np.array([[1.0, 1.0, 1.0], [6.0, 1.0, 1.0]], np.float32)


def _two(units=(None, None)):
    return [td.uniform_grid((3, 3, 3), unit=units[0]),
            td.uniform_grid((3, 3, 3), origin=(5.0, 0.0, 0.0), unit=units[1])]


def _with(piece, name, value, unit=None, cells=False):
    n = piece.num_cells if cells else piece.num_points
    space = td.Constant(piece) if cells else td.H1(piece)
    piece.fields[name] = td.Field(space, tack.array(np.full(n, value, np.float32)), unit)
    return piece


def test_probe_refuses_pieces_that_give_back_different_fields(backend):
    a, b = _two()
    pieces = td.PartitionedDataSet([_with(a, "a", 1.0), _with(b, "b", 1.0)])
    with pytest.raises(ValueError, match="same fields"):
        td.probe(pieces, _POINTS)
    # Choosing a field both have probes it.
    _with(a, "c", 1.0)
    _with(b, "c", 2.0)
    assert td.probe(pieces, _POINTS, fields=["c"]).fields["c"].values.numpy().tolist() == [1, 2]


def test_probe_refuses_and_agrees_on_units(backend):
    a, b = _two()
    mixed = td.PartitionedDataSet([_with(a, "v", 1.0, "m/s"), _with(b, "v", 100.0, "cm/s")])
    with pytest.raises(ValueError, match="'m/s' and 'cm/s'"):
        td.probe(mixed, _POINTS, fields=["v"])
    a, b = _two()
    alike = td.PartitionedDataSet([_with(a, "v", 1.0, "m/s"), _with(b, "v", 2.0, "m/s")])
    assert td.probe(alike, _POINTS).fields["v"].unit == "m/s"
    a, b = _two()
    unknown = td.PartitionedDataSet([_with(a, "v", 1.0, "m/s"), _with(b, "v", 2.0)])
    assert td.probe(unknown, _POINTS).fields["v"].unit is None     # as units.common has it


def test_integrate_refuses_pieces_in_two_units(backend):
    with pytest.raises(ValueError, match="'m\\*\\*3' and 'cm\\*\\*3'"):
        td.integrate(td.PartitionedDataSet(_two(("m", "cm"))))
    a, b = _two(("m", "m"))
    with pytest.raises(ValueError, match="field 't' is in 'K' and"):
        td.integrate(td.PartitionedDataSet([_with(a, "t", 1.0, "K"), _with(b, "t", 1.0, "degC")]))
    a, b = _two(("m", "m"))
    a.fields["only"] = a.fields["t"] = td.Field(td.H1(a), tack.zeros((27,)))
    b.fields["t"] = td.Field(td.H1(b), tack.zeros((27,)))
    with pytest.raises(ValueError, match="same fields"):
        td.integrate(td.PartitionedDataSet([a, b]))


def test_pieces_integrate_to_the_whole(backend):
    whole = td.uniform_grid((9, 7, 5), spacing=(0.5, 0.25, 1.0), unit="m")
    x = whole.geometry.values.numpy().astype(float)
    whole.fields["s"] = td.Field(td.H1(whole), tack.array((x[:, 0] + x[:, 1] * x[:, 2])
                                                          .astype(np.float32)), "K")
    pieces = td.split(whole, (2, 2, 1))
    a, b = td.integrate(whole), td.integrate(pieces)
    assert (a.measure_unit, a.units) == (b.measure_unit, b.units) == ("m**3", {"s": "K*m**3"})
    assert math.isclose(a.measure, b.measure, rel_tol=1e-12)
    assert math.isclose(a.values["s"], b.values["s"], rel_tol=1e-6)


def test_merge_pieces_refuses_and_agrees_on_units(backend):
    pieces = td.split(td.uniform_grid((5, 3, 3)), (2, 1, 1))
    for piece, unit in zip(pieces, ("m", "cm")):
        _with(piece, "q", 1.0, unit, cells=True)
    with pytest.raises(ValueError, match="'m' and 'cm'"):
        td.merge_pieces(pieces, fields=["q"])
    for piece in pieces:
        _with(piece, "q", 1.0, "m", cells=True)
    assert td.merge_pieces(pieces, fields=["q"]).fields["q"].unit == "m"
    _with(pieces[1], "q", 1.0, None, cells=True)
    assert td.merge_pieces(pieces, fields=["q"]).fields["q"].unit is None


def test_ghost_cells_refuse_pieces_in_two_units(backend):
    pieces = td.split(td.uniform_grid((5, 3, 3)), (2, 1, 1))
    for piece, unit in zip(pieces, ("m", "cm")):
        _with(piece, "q", 1.0, unit, cells=True)
    with pytest.raises(ValueError, match="'m' and 'cm'"):
        td.ghost_cells(pieces, fields=["q"])
