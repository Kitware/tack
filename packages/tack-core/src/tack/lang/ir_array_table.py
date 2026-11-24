# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""IR pass: a kernel's array pointers in one table, when they would not fit.

A device limits the total size of a kernel's arguments: 2 KB on the Intel
GPUs Level Zero drives, 4 KB for CUDA and HIP. Scalars already go in packs
(``ir_pack_scalars``), so what remains is one 8-byte pointer per array, and
a kernel over enough arrays (device-class objects flatten to many) could
not be built at all. A variant past the limit takes instead one u64 array,
the table, whose entries are its arrays' addresses; the code generators
declare each array as a pointer read from its entry at kernel entry
(``CUDACodeGen``, ``OpenCLCodeGen``), and dispatch writes the table before
each launch, as it writes the scalar packs (``GPUBackend._array_table``).

The marked parameters stay in ``params``, so every analysis and the GPU
renaming see the IR they always did; only the kernel's signature and its
arguments leave them out (`launch_params`).
"""

from tack.lang import ir
from tack.lang.ir_names import fresh_name, ir_names
from tack.lang.types import u64

#: Bytes of launch arguments every kernel takes besides its parameters:
#: ``__n__``, and for a multi-dimensional loop ``__flat__`` and an extent
#: per dimension.
_LAUNCH_BYTES = 8 + 4 * 8


def argument_bytes(ir_func: ir.IRFunction) -> int:
    """An upper bound on the size of a packed kernel's arguments: every
    parameter a pointer or 8-byte value, and the launch arguments."""
    return 8 * len(launch_params(ir_func)) + _LAUNCH_BYTES


def table_arrays(ir_func: ir.IRFunction) -> tuple:
    """Mark every array parameter (but hardware textures) as an entry of a new
    table parameter, appended to ``params``. Returns the marked parameters'
    positions in ``params``, in table order: the arguments whose addresses
    the table holds."""
    positions = []
    for index, param in enumerate(ir_func.params):
        if param._is_array is not False and not param._is_texture:
            param._table_slot = len(positions)
            positions.append(index)
    table = ir.IRParam(fresh_name("__tack_arrays__", ir_names(ir_func)), type_annotation=u64)
    table._is_array = True
    table._is_array_table = True
    ir_func.params.append(table)
    return tuple(positions)


def launch_params(ir_func: ir.IRFunction) -> list:
    """The parameters a launch passes: all but the arrays read from the table."""
    return [p for p in ir_func.params if p._table_slot is None]


def table_entries(ir_func: ir.IRFunction) -> tuple:
    """``(table, [(slot, param)...])``: the table parameter and the arrays it
    holds, for a code generator; ``(None, [])`` without a table."""
    table = next((p for p in ir_func.params if p._is_array_table), None)
    entries = sorted((p._table_slot, p) for p in ir_func.params if p._table_slot is not None)
    return table, entries
