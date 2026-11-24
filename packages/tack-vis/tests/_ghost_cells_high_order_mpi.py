# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""``test_ghost_cells_high_order``'s check across MPI ranks: ``mpiexec -n 2 python
_ghost_cells_high_order_mpi.py``. Every rank grows its share of six pieces, loses
the values its own cells do not have and refreshes them; rank 0 compares each
piece with what one process makes of all of them, and prints ``ok`` or fails."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import test_ghost_cells_high_order as case
from mpi4py import MPI

import tack
from tack import communication as tc


def main():
    tack.set_default_device("cpu")
    comm = MPI.COMM_WORLD
    for structured in (False, True):
        mine = case.refreshed_across(tc.World(comm), structured)
        gathered = comm.gather(mine, root=0)
        if comm.rank == 0:
            alone = case.refreshed_across(tc.World(), structured)
            across = {g: d for part in gathered for g, d in part.items()}
            assert sorted(across) == sorted(alone)
            for g in alone:
                for name, values in alone[g].items():
                    assert not np.isnan(values).any(), (g, name)
                    np.testing.assert_array_equal(across[g][name], values,
                                                  err_msg=f"piece {g}, {name}")
    if comm.rank == 0:
        print(f"ok: 6 pieces on {comm.size} ranks, unstructured and structured", flush=True)


if __name__ == "__main__":
    main()
