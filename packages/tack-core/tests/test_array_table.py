# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A kernel's arrays read from one table when its arguments would not fit.

A device limits the total size of a kernel's arguments (2 KB on Level
Zero's Intel GPUs, 4 KB for CUDA and HIP); scalars go in packs, and past
the limit the arrays' addresses go in a u64 table the variant keeps and
each launch rewrites (`ir_array_table`, `GPUBackend._array_table`), read
back into typed pointers at kernel entry. The first tests check the
generated CUDA, HIP and OpenCL on the host; the rest run on the GPU
backends with a limit small enough that every kernel takes a table.
"""

import subprocess
import threading

import numpy as np
import pytest
from compiler_tools import require_clang

import tack
from tack.codegen.cuda_gen import generate_cuda_source
from tack.codegen.hip_gen import generate_hip_source
from tack.codegen.identifiers import kernel_entry_name
from tack.codegen.opencl_gen import generate_opencl_source
from tack.lang.inspect_kernel import _prepare_ir
from tack.lang.ir_array_table import argument_bytes, launch_params, table_arrays, table_entries
from tack.lang.ir_pack_scalars import pack_scalars
from tack.lang.ir_type_annotate import annotate_types
from tack.runtime import kernel_utils
from tack.runtime.dispatch import get_backend

pytestmark = pytest.mark.quick


@tack.kernel
def _blend(a, b, c, out):
    for i in range(out.shape[0]):
        out[i] = a[i] + 2.0 * b[i] * c[i]


@tack.kernel
def _scaled(a, b, out, s, k):
    for i in range(out.shape[0]):
        out[i] = s * a[i] + b[(i + k) % b.shape[0]]


@tack.kernel
def _counts(keys, counts):
    for i in range(keys.shape[0]):
        tack.atomic_add(counts, keys[i], 1)


@tack.kernel
def _grid(a, b, out):
    for i, j in tack.ndindex(out.shape[0], out.shape[1]):
        out[i, j] = a[i, j] - b[i, j]


@tack.kernel
def _norms(p, out):
    for i in range(out.shape[0]):
        out[i] = p[i].norm()


@tack.device_class
class _Component:
    def __init__(self, values, dofs):
        self.values, self.dofs = values, dofs

    @tack.func
    def at(self, i):
        return self.values[self.dofs[i]]


@tack.kernel
def _combine(a, b, c, out):
    for i in range(out.shape[0]):
        out[i] = a.at(i) + 2.0 * b.at(i) + 4.0 * c.at(i)


def _tabled_ir(kernel, args):
    """A kernel's IR as a GPU backend compiles it, with every array in the table."""
    function, effective = _prepare_ir(kernel, args)
    pack_scalars(function, effective)
    annotate_types(function)
    table_arrays(function)
    return function


# ── The generated code, on the host ────────────────────────────────


@pytest.mark.usefixtures("no_debug")
def test_the_table_takes_the_arrays_out_of_the_signature():
    tack.set_default_device("cpu")
    args = (tack.zeros((4,)), tack.zeros((4,)), tack.zeros((4,)), 1.5, 2)
    function, effective = _prepare_ir(_scaled, args)
    pack_scalars(function, effective)
    annotate_types(function)
    before = argument_bytes(function)
    positions = table_arrays(function)
    # a, b, out and the two scalar packs (f32, i32), in parameter order.
    assert positions == (0, 1, 2, 3, 4)
    table, entries = table_entries(function)
    assert [slot for slot, _ in entries] == [0, 1, 2, 3, 4]
    assert launch_params(function) == [table]
    assert argument_bytes(function) < before
    source = generate_cuda_source(function)
    entry = next(line for line in source.splitlines() if "__global__ void" in line)
    assert entry.count("*") == 1, entry               # the table alone
    assert source.count(f"[{len(positions) - 1}];") >= 1


@pytest.mark.parametrize("generate", [generate_cuda_source, generate_hip_source],
                         ids=["cuda-host", "hip-host"])
@pytest.mark.usefixtures("no_debug")
def test_generated_code_reads_its_arrays_from_the_table(generate, tmp_path):
    clang = require_clang("ubsan")
    tack.set_default_device("cpu")
    n = 8
    rng = np.random.default_rng(1)
    inputs = [rng.random(n).astype(np.float32) for _ in range(3)]
    function = _tabled_ir(_blend, (*map(tack.array, inputs), tack.zeros((n,))))
    _, entries = table_entries(function)
    names = [param.name for _, param in entries]
    assert names == ["a", "b", "c", "out"]
    source = "\n".join(line for line in generate(function).splitlines()
                       if not line.startswith("#include"))
    table = ", ".join(f"(unsigned long long)h_{name}" for name in names)
    program = """
#include <cmath>
#include <cstdio>
#define __global__
#define __device__
static inline float __fmul_rn(float a, float b) { return a * b; }
static inline float __fmaf_rn(float a, float b, float c) { return std::fma(a, b, c); }
struct Grid { unsigned int x; };
Grid blockIdx = {0}, blockDim = {1}, threadIdx = {0};
""" + source + f"""
int main(int argc, char** argv) {{
    float h_a[{n}], h_b[{n}], h_c[{n}], h_out[{n}];
    FILE* input = std::fopen(argv[1], "rb");
    if (!input) return 2;
    if (std::fread(h_a, sizeof(float), {n}, input) != {n}) return 3;
    if (std::fread(h_b, sizeof(float), {n}, input) != {n}) return 3;
    if (std::fread(h_c, sizeof(float), {n}, input) != {n}) return 3;
    std::fclose(input);
    unsigned long long table[] = {{{table}}};
    for (unsigned int i = 0; i < {n}; ++i) {{
        blockIdx.x = i;
        {kernel_entry_name(function.name)}(table, {n});
    }}
    return std::fwrite(h_out, sizeof(float), {n}, stdout) == {n} ? 0 : 4;
}}
"""
    cpp, executable, data = tmp_path / "kernel.cpp", tmp_path / "kernel", tmp_path / "in.bin"
    cpp.write_text(program)
    result = subprocess.run([clang, "-std=c++14", "-O2", "-fsanitize=undefined",
                             "-fno-sanitize-recover=all", str(cpp), "-o", str(executable)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr + program
    np.concatenate(inputs).tofile(data)
    result = subprocess.run([str(executable), str(data)], capture_output=True)
    assert result.returncode == 0, result.stderr.decode()
    got = np.frombuffer(result.stdout, dtype=np.float32)
    np.testing.assert_allclose(got, inputs[0] + 2 * inputs[1] * inputs[2], rtol=1e-6)


@pytest.mark.usefixtures("no_debug")
def test_generated_opencl_with_a_table_compiles(tmp_path):
    clang = require_clang()
    tack.set_default_device("cpu")
    for kernel, args in (
            (_scaled, (tack.zeros((4,)), tack.zeros((4,)), tack.zeros((4,)), 1.5, 2)),
            (_counts, (tack.zeros((4,), tack.i32), tack.zeros((4,), tack.i32))),
            (_grid, (tack.zeros((2, 3)), tack.zeros((2, 3)), tack.zeros((2, 3))))):
        source = generate_opencl_source(_tabled_ir(kernel, args))
        assert "__global unsigned long*" in source
        path = tmp_path / f"{kernel.name}.cl"
        path.write_text(source)
        result = subprocess.run([clang, "-target", "x86_64-unknown-linux-gnu", "-x", "cl",
                                 "-cl-std=CL2.0", "-fsyntax-only", str(path)],
                                capture_output=True, text=True)
        assert result.returncode == 0, result.stderr + source


# ── On the GPU backends ────────────────────────────────────────────


@pytest.fixture
def tables(backend, monkeypatch):
    """A GPU backend whose every kernel takes an array table: its argument
    limit below any kernel's, for kernels compiled in the test."""
    device = get_backend()
    if device.name not in ("cuda", "hip", "level_zero"):
        pytest.skip(f"{device.label} has no limit on its kernels' arguments")
    monkeypatch.setattr(device, "_max_argument_bytes", 16)
    yield device


def _variants(kernel):
    return list(kernel_utils.kernel_cache_slot(get_backend()._cache, kernel).values())


def _took_a_table(kernel):
    assert _variants(kernel)
    assert all(v.payload[3] is not None for v in _variants(kernel))


def test_arrays_and_scalars_through_a_table(tables):
    n = 300
    rng = np.random.default_rng(2)
    a, b = (rng.random(n).astype(np.float32) for _ in range(2))
    longer = tack.array(np.concatenate([np.zeros(5, np.float32), b]))
    out = tack.zeros((n,))
    _scaled(tack.array(a), longer[5:], out, 1.5, 7)      # a view: its own address
    np.testing.assert_allclose(out.numpy(), 1.5 * a + np.roll(b, -7), rtol=1e-6)
    # Other arrays through the same variant: the table is rewritten per launch.
    c = rng.random(n).astype(np.float32)
    _scaled(tack.array(c), tack.array(a), out, 2.0, 1)
    np.testing.assert_allclose(out.numpy(), 2.0 * c + np.roll(a, -1), rtol=1e-6)
    _took_a_table(_scaled)


def test_launches_queued_back_to_back_keep_their_own_arrays(tables):
    n = 1000
    outs = [tack.zeros((n,)) for _ in range(8)]
    inputs = [tack.array(np.full(n, k, np.float32)) for k in range(8)]
    for k in range(8):            # queued: each launch must read its own table
        _blend(inputs[k], inputs[k], inputs[k], outs[k])
    for k in range(8):
        np.testing.assert_allclose(outs[k].numpy(), k + 2.0 * k * k)


def test_atomics_on_a_tabled_array(tables):
    keys = np.random.default_rng(3).integers(0, 16, 5000).astype(np.int32)
    counts = tack.zeros((16,), tack.i32)
    _counts(tack.array(keys), counts)
    np.testing.assert_array_equal(counts.numpy(), np.bincount(keys, minlength=16))
    _took_a_table(_counts)


def test_a_multidimensional_launch_through_a_table(tables):
    a = np.arange(35, dtype=np.float32).reshape(5, 7)
    out = tack.zeros((5, 7))
    _grid(tack.array(a), tack.array(2 * a), out)
    np.testing.assert_allclose(out.numpy(), -a)
    _took_a_table(_grid)


def test_vector_and_soa_arrays_through_a_table(tables):
    p = np.random.default_rng(4).random((50, 3)).astype(np.float32)
    for layout in ("aos", "soa"):
        out = tack.zeros((50,))
        _norms(tack.array(p, dtype=tack.vec3(tack.f32), layout=layout), out)
        np.testing.assert_allclose(out.numpy(), np.linalg.norm(p, axis=1), rtol=1e-5)


def test_shared_template_arrays_through_a_table(tables):
    n = 64
    rng = np.random.default_rng(5)
    dofs = tack.array(rng.permutation(n).astype(np.int32))
    values = [rng.random(n).astype(np.float32) for _ in range(3)]
    out = tack.zeros((n,))
    _combine(*(_Component(tack.array(v), dofs) for v in values), out)
    d = dofs.numpy()
    np.testing.assert_allclose(out.numpy(), values[0][d] + 2 * values[1][d] + 4 * values[2][d],
                               rtol=1e-6)
    _took_a_table(_combine)


def test_debug_mode_through_a_table(tables):
    n = 20
    a = tack.array(np.arange(n, dtype=np.float32))
    out = tack.zeros((n,))
    was = tack.runtime.dispatch.debug_enabled()
    tack.set_debug(True)
    try:
        _blend(a, a, a, out)
        with pytest.raises(IndexError):
            _blend(a, a, a, tack.zeros((n + 5,)))        # reads past a, b and c
    finally:
        tack.set_debug(was)
    np.testing.assert_allclose(out.numpy(), np.arange(n) + 2.0 * np.arange(n) ** 2)


def test_threads_launching_one_tabled_variant(tables):
    n = 4096
    _blend(*(tack.zeros((n,)) for _ in range(4)))         # compile it once
    errors = []

    def run(k):
        try:
            x = tack.array(np.full(n, k, np.float32))
            out = tack.zeros((n,))
            for _ in range(20):
                _blend(x, x, x, out)
            np.testing.assert_allclose(out.numpy(), k + 2.0 * k * k)
        except Exception as e:          # reported below, on the test's thread
            errors.append(e)

    if tables.name == "cuda":
        pytest.skip("CUDA's context is current only on the thread that made the device")
    threads = [threading.Thread(target=run, args=(k,)) for k in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
