# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""``test_temporal``'s check under MPI: ``mpiexec -n 2 python _temporal_mpi.py``. The
steps' pieces are shared out over the ranks, each rank advects its own seeds
through them (each particle carried by the piece that holds it, wherever its
rank), and its particles must be where advection through the whole steps puts
them, on two and four pieces, with ghost cells and without; rank 0 prints
``ok``."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import test_temporal as case
from mpi4py import MPI

from tack import communication as tc
from tack.dataset import temporal


def main():
    comm = MPI.COMM_WORLD
    # The pieces are on one mesh: each piece's kernel carries the particles it
    # holds, whatever their rank.
    rounds = []
    real = temporal.Advection._advance_pieces
    temporal.Advection._advance_pieces = lambda self, *a: (rounds.append(1), real(self, *a))
    for uneven in (False, True):
        for cuts, ghosts in (((2, 2, 1), False), ((2, 2, 1), True), ((2, 1, 1), False)):
            got, want = case.advect_across(tc.World(comm), uneven, cuts, ghosts)
            np.testing.assert_allclose(got.world_positions(), want.world_positions(),
                                       atol=1e-5)
            np.testing.assert_array_equal(got.fields["status"].values.numpy(),
                                          want.fields["status"].values.numpy())
    assert len(rounds) == 12, rounds
    done = comm.gather(True, root=0)
    if comm.rank == 0 and all(done):
        print(f"ok: {comm.size} ranks", flush=True)


if __name__ == "__main__":
    main()
