# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Regression for overlapping row views in PR #393's NumPy-style copyto API."""

import numpy as np
import pytest

import tack


@pytest.mark.parametrize("layout", ["aos", "soa"])
@pytest.mark.parametrize("direction", ["right", "left"])
@pytest.mark.parametrize("snapshot_source", [False, True], ids=["overlap", "copied-source"])
def test_copyto_overlapping_rows(layout, direction, snapshot_source):
    # One CPU worker makes the right-shift failure deterministic rather than
    # depending on a GPU's scheduling or a race between CPU workers.
    device = tack.device("cpu", num_threads=1)
    host = np.arange(32, dtype=np.int32).reshape(16, 2)
    values = tack.array(host, tack.vec2(tack.i32), device=device, layout=layout)
    dst_rows, src_rows = ((slice(1, None), slice(None, -1)) if direction == "right"
                          else (slice(None, -1), slice(1, None)))
    expected = host.copy()
    np.copyto(expected[dst_rows], expected[src_rows])

    source = values[src_rows]
    if snapshot_source:
        # Control: the exact same operation with independently stored input.
        source = source.copy()
    tack.copyto(values[dst_rows], source)

    np.testing.assert_array_equal(values.numpy(), expected)
