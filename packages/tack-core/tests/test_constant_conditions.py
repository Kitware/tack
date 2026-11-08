# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Branches that constants decide are resolved while lowering
(lang/lowering/constant_conditions.py): only the arm taken is lowered."""

import numpy as np
import pytest

import tack

pytestmark = pytest.mark.quick      # the inner-loop set; see CLAUDE.md, Tests

NOT_A_CONSTANT = 0.5                # a module float: a kernel reading it fails to lower


def _branches(kernel, *args):
    """The branches left in the kernel's IR for these arguments."""
    text = tack.inspect(kernel, *args, mode="ir")
    return [line for line in text.splitlines() if "If (" in line or "IfExp(" in line]


@tack.device_class
class Shape:
    SIDES = 3

    @tack.func
    def sides(self):
        if self.SIDES == 3:
            return 30
        if self.SIDES == 4:
            return 40
        return 0


@tack.device_class
class Square(Shape):
    SIDES = 4


@tack.kernel
def by_class(shape, out):
    for i in range(out.shape[0]):
        out[i] = shape.sides()


@pytest.fixture(autouse=True)
def _cpu():
    tack.set_default_device("cpu")


@pytest.mark.parametrize("shape, expected", [(Shape(), 30), (Square(), 40)])
def test_a_class_constant_picks_the_arm(shape, expected):
    out = tack.zeros((4,), tack.i32)
    by_class(shape, out)
    np.testing.assert_array_equal(out.numpy(), np.full(4, expected))
    assert _branches(by_class, shape, out) == []


@tack.kernel
def unlowerable_arm(out):
    for i in range(out.shape[0]):
        if 0 == 1:
            out[i] = NOT_A_CONSTANT      # would raise NameError if lowered
        else:
            out[i] = 7


def test_the_arm_not_taken_is_not_lowered():
    out = tack.zeros((3,))
    unlowerable_arm(out)
    np.testing.assert_array_equal(out.numpy(), np.full(3, 7))


@tack.kernel
def barrier_never_run(out):
    for i in range(out.shape[0]):
        if 1 > 2:
            tack.barrier()
        out[i] = i


def test_a_primitive_in_an_arm_not_taken_is_not_required():
    out = tack.zeros((4,), tack.i32)
    barrier_never_run(out)              # the CPU has no workgroups
    np.testing.assert_array_equal(out.numpy(), np.arange(4))


@tack.kernel
def false_and_runtime(data, out):
    for i in range(out.shape[0]):
        if (6 & 1) != 0 and data[i] > 0:
            out[i] = 1
        else:
            out[i] = 2


@tack.kernel
def true_or_runtime(data, out):
    for i in range(out.shape[0]):
        out[i] = 1 if data[i] > 0 or not False else 2


def test_and_or_are_decided_by_one_operand():
    data = tack.array(np.array([1.0, -1.0], np.float32))
    out = tack.zeros((2,), tack.i32)
    false_and_runtime(data, out)
    np.testing.assert_array_equal(out.numpy(), [2, 2])
    true_or_runtime(data, out)
    np.testing.assert_array_equal(out.numpy(), [1, 1])
    assert _branches(false_and_runtime, data, out) == []
    assert _branches(true_or_runtime, data, out) == []


@tack.kernel
def runtime_branch(data, out):
    for i in range(out.shape[0]):
        if 0.5 < 1.0 and data[i] > 0:
            out[i] = 1
        else:
            out[i] = 2


def test_float_and_runtime_conditions_are_left_to_run():
    data = tack.array(np.array([1.0, -1.0], np.float32))
    out = tack.zeros((2,), tack.i32)
    runtime_branch(data, out)
    np.testing.assert_array_equal(out.numpy(), [1, 2])
    assert len(_branches(runtime_branch, data, out)) == 1


@tack.kernel
def bound_in_the_arm_not_taken(out):
    for i in range(out.shape[0]):
        if 1 == 2:
            x = 1
        out[i] = x


def test_a_name_bound_only_in_the_arm_not_taken_is_unbound():
    with pytest.raises(Exception, match="x"):
        bound_in_the_arm_not_taken(tack.zeros((2,), tack.i32))
