# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""New arrays filled with a byte (``TACK_FILL_NEW``, or 0xFF in debug mode), on
every backend: no backend clears a new buffer, since `tack.empty`'s contents are
unspecified, so the fill is how a read before a write shows (#386)."""

import numpy as np
import pytest

import tack
from tack.lang import array as array_module
from tack.runtime import dispatch

pytestmark = pytest.mark.quick


@pytest.fixture
def fill(monkeypatch):
    """New arrays filled with 0xAB for the test."""
    monkeypatch.setattr(array_module, "_FILL_NEW", 0xAB)
    return 0xAB


def _bytes(a):
    values = a.numpy()
    return np.ascontiguousarray(values).view(np.uint8)


@pytest.mark.parametrize("dtype", ["f32", "i32", "u8", "i64", "vec3"])
def test_new_arrays_are_filled_with_the_byte_on_every_backend(backend, fill, dtype):
    element = {"f32": tack.f32, "i32": tack.i32, "u8": tack.u8, "i64": tack.i64,
               "vec3": tack.vec3(tack.f32)}[dtype]
    a = tack.empty((1000,), element)
    assert (_bytes(a) == fill).all()
    assert tack.empty((0,), element).numpy().size == 0


def test_zeros_ones_and_full_still_hold_their_values(backend, fill):
    """`zeros` fills after the byte, Metal's fresh buffers included (which `zeros`
    otherwise takes as Metal cleared them, ``starts_zeroed``)."""
    assert (tack.zeros((1 << 18,), tack.f32).numpy() == 0).all()
    assert (tack.zeros((300,), tack.i32).numpy() == 0).all()
    assert (tack.ones((300,), tack.i32).numpy() == 1).all()
    assert (tack.full((300,), 7.5, tack.f32).numpy() == 7.5).all()


def test_a_filled_float_reads_as_nan_and_an_integer_as_minus_one_or_its_largest(
        backend, monkeypatch):
    monkeypatch.setattr(array_module, "_FILL_NEW", 0xFF)
    assert np.isnan(tack.empty((64,), tack.f32).numpy()).all()
    assert (tack.empty((64,), tack.i32).numpy() == -1).all()
    assert (tack.empty((64,), tack.u8).numpy() == 255).all()
    assert (tack.empty((64,), tack.u32).numpy() == 2**32 - 1).all()


def test_structure_of_arrays_and_mpi_arrays_are_filled(fill):
    soa = tack.empty((50,), tack.vec3(tack.f32), layout="soa")
    assert all((_bytes(c) == fill).all() for c in soa.components)
    received = array_module._empty_for_mpi((40,), tack.i32, None)
    assert (_bytes(received) == fill).all()


def test_debug_mode_fills_with_0xff_unless_a_byte_is_set(monkeypatch):
    monkeypatch.setattr(array_module, "_FILL_NEW", None)
    monkeypatch.setattr(dispatch, "_debug", True)
    assert (tack.empty((16,), tack.i32).numpy() == -1).all()
    monkeypatch.setattr(array_module, "_FILL_NEW", 0x11)
    assert (_bytes(tack.empty((16,), tack.i32)) == 0x11).all()
    monkeypatch.setattr(array_module, "_FILL_NEW", None)
    monkeypatch.setattr(dispatch, "_debug", False)
    assert array_module._new_fill() is None


@pytest.mark.parametrize("text, byte", [(None, None), ("", None), ("off", "off"), (" OFF ", "off"),
                                        ("0xFF", 255), ("255", 255), ("0", 0), ("0xab", 0xAB)])
def test_the_setting_is_a_byte_or_off(text, byte):
    assert array_module._fill_setting(text) == byte


def test_off_turns_the_fill_off_in_debug_mode_too(monkeypatch):
    """``TACK_FILL_NEW=off`` keeps debug mode's bounds checks and drops its fill;
    unset, debug mode fills. In a process of its own, as the variable is read when
    Tack is imported."""
    import os
    import subprocess
    import sys

    probe = ("import tack; from tack.lang import array; from tack.runtime import dispatch; "
             "print(dispatch.debug_enabled(), array._new_fill())")
    for setting, expected in (("off", "True None"), ("", "True 255")):
        env = {**os.environ, "TACK_DEBUG": "1", "TACK_FILL_NEW": setting}
        run = subprocess.run([sys.executable, "-c", probe], env=env, capture_output=True,
                             text=True)
        assert run.returncode == 0, run.stderr
        assert run.stdout.split() == expected.split(), (setting, run.stdout)
    monkeypatch.setattr(array_module, "_FILL_NEW", array_module._FILL_OFF)
    monkeypatch.setattr(dispatch, "_debug", True)
    assert array_module._new_fill() is None

    @tack.kernel
    def past_the_end(a):
        for i in range(a.shape[0]):
            a[i + 1] = 1.0

    with pytest.raises(IndexError):
        past_the_end(tack.zeros((4,), tack.f32))


@pytest.mark.parametrize("text", ["256", "-1", "on", "0x100", "nan"])
def test_the_setting_refuses_anything_else(text):
    with pytest.raises(ValueError, match="TACK_FILL_NEW"):
        array_module._fill_setting(text)
