# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Comparisons and small datasets shared by the tack_expr tests."""

import numpy as np

import tack
import tack.dataset as td


def assert_close(got, ref, tol=1e-4):
    """Each value within ``tol`` of the reference, relative where it exceeds 1."""
    ref = np.asarray(ref, np.float64)
    got = np.asarray(got, np.float64).reshape(ref.shape)
    err = float(np.max(np.abs(got - ref) / np.maximum(1.0, np.abs(ref))))
    assert err < tol, f"max relative error {err:.1e} (tolerance {tol:.0e})"


def assert_close_to_largest(got, ref, tol, mask=None):
    """Every value within ``tol`` of the reference's largest magnitude (or 1)."""
    ref = np.asarray(ref, np.float64)
    got = np.asarray(got, np.float64).reshape(ref.shape)
    if mask is not None:
        got, ref = got[mask], ref[mask]
    err = float(np.max(np.abs(got - ref)) / max(1.0, float(np.max(np.abs(ref)))))
    assert err < tol, f"max error {err:.1e} of the largest value (tolerance {tol:.0e})"


def values(expr, data, at="cells", **kwargs):
    import tack_expr as fx

    return fx.evaluate(expr, data, at=at, **kwargs).values.numpy()


def vector_array(rows):
    v = tack.empty((len(rows),), tack.vec3())
    tack.copyto(v, np.ascontiguousarray(rows, dtype=np.float32))
    return v


def with_fields(data, rng, space):
    """Random scalars T and P and a vector V in ``space`` (``H1`` or ``Constant``)."""
    s = space(data)
    n = s.size
    data.fields["T"] = td.Field(s, tack.array(rng.random(n, dtype=np.float32)))
    data.fields["P"] = td.Field(s, tack.array(rng.random(n, dtype=np.float32)))
    data.fields["V"] = td.Field(s, vector_array(rng.random((n, 3), dtype=np.float32) * 2 - 1))
    return data


def position_flux(data):
    """Each face's integrated flux of v = (x, y, z), whose divergence is 3."""
    geometry = td.face_geometry(data)
    centers = td.face_centers(data).values.numpy()
    flux = np.sum(centers * geometry.normal.values.numpy(), axis=1) * geometry.area.values.numpy()
    return flux.astype(np.float32)


def with_position_flux(data):
    data.fields["flux"] = td.Field(td.Values(data, "faces", oriented=True), tack.array(position_flux(data)))
    return data
