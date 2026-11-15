# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Pre-pass kernels: one component of a vector array, and the cell of each point link."""

import tack


@tack.kernel
def component(values, k, out, n):
    for i in range(n):
        out[i] = values[i][k]


@tack.kernel
def entry_cells(cells, entries, out, n, start, end):
    """The cell of each link entry in [start, end): one cell group's corners in the
    topology's corner layout, NUM_POINTS per cell."""
    for i in range(n):
        e = entries[i]
        if e >= start and e < end:
            out[i] = cells.entity_id((e - start) // cells.NUM_POINTS)


@tack.kernel
def position_cells(offsets, out, n):
    """The cell of each position in a CSR of per-cell lists."""
    for c in range(n):
        for e in range(offsets[c], offsets[c + 1]):
            out[e] = c


@tack.kernel
def gather(source, index, out, n):
    for i in range(n):
        out[i] = source[index[i]]

