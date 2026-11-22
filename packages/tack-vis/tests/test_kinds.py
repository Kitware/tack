# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A field's kind (``Field.kind``, ``tack.dataset.kinds``): filled from the registry
by name, each part on its own without one, checked against the field's unit and
values, and kept where a field is moved or converted
(``docs/design/plans/field-quantities.md``, D1 and D5)."""

import pytest

import tack
import tack.dataset as td
from tack.dataset import kinds
from tack.runtime import dispatch
from tack.runtime.cpu import CPUBackend

pytestmark = pytest.mark.quick


class _SecondCPU(CPUBackend):
    """A CPU backend posing as another device's: its arrays are not the default's."""

    display_name = "second device"

    @property
    def memory_domain(self):
        return ("cpu", 1)


@pytest.fixture
def grid():
    return td.uniform_grid((3, 3, 3))


def scalar(data, unit="Pa", kind="Pressure"):
    return td.Field(td.H1(data), tack.zeros((data.num_points,)), unit, kind=kind)


def test_a_name_fills_the_parts_from_the_registry():
    k = td.Kind("Vorticity")
    assert (k.rank, k.extent, k.parity, k.variance) == ("vector", "intensive", "axial",
                                                        "covariant")
    assert td.Kind("Stress").rank == "symmetric tensor"
    assert td.Kind("Density", rank="scalar") == td.Kind("Density")


def test_given_parts_must_agree_with_the_name():
    with pytest.raises(ValueError, match="axial"):
        td.Kind("Vorticity", parity="polar")
    with pytest.raises(ValueError, match="vector"):
        td.Kind("Velocity", rank="scalar")
    with pytest.raises(KeyError, match="registry"):
        td.Kind("Flux capacitance")
    with pytest.raises(ValueError, match="a Pressure .a scalar. has no variance"):
        td.Kind("Pressure", variance="covariant")
    with pytest.raises(ValueError, match="a Mach .a scalar. has no variance"):
        td.Kind("Mach", variance="mixed")
    with pytest.raises(ValueError, match="only a tensor is mixed"):
        td.Kind("HeatFlux", variance="mixed")


def test_each_part_stands_alone_without_a_name():
    assert td.Kind(rank="vector").name is None
    assert td.Kind(extent="extensive").rank is None
    assert td.Kind() == td.Kind()
    assert repr(td.Kind(rank="vector")) == "Kind(rank='vector')"


@pytest.mark.parametrize("parts, message", [
    ({"rank": "matrix"}, "rank"),
    ({"extent": "massive"}, "extent"),
    ({"parity": "odd"}, "parity"),
    ({"variance": "sideways"}, "variance"),
    ({"rank": "scalar", "variance": "covariant"}, "no variance"),
    ({"rank": "vector", "variance": "mixed"}, "only a tensor"),
])
def test_parts_take_their_values(parts, message):
    with pytest.raises(ValueError, match=message):
        td.Kind(**parts)


def test_kinds_compare_by_value_and_hash():
    assert td.Kind("Pressure") == td.Kind("Pressure")
    assert td.Kind("Pressure") != td.Kind("Stress")
    assert len({td.Kind("Pressure"), td.Kind("Pressure"), td.Kind(rank="vector")}) == 2


def test_a_field_takes_a_kind_or_a_name(grid):
    p = scalar(grid, "kPa")
    assert p.kind == td.Kind("Pressure") and p.unit == "kPa"
    assert scalar(grid, kind=td.Kind("Pressure")).kind == p.kind
    assert scalar(grid, kind=None).kind is None
    assert "Pressure" in repr(p)
    with pytest.raises(TypeError, match="Kind"):
        scalar(grid, kind=3)


def test_the_unit_must_have_the_kinds_dimensions(grid):
    assert scalar(grid, "psi").kind.name == "Pressure"
    assert scalar(grid, None).kind.name == "Pressure"         # an unknown unit is not checked
    with pytest.raises(ValueError, match="units like 'Pa'"):
        scalar(grid, "m/s")
    assert scalar(grid, "m/s", td.Kind(rank="scalar")).kind.rank == "scalar"


def test_the_values_must_fit_the_rank(grid):
    n = grid.num_points
    vectors = tack.zeros((n,), tack.vec3())
    assert td.Field(td.H1(grid), vectors, "m/s", kind="Velocity").kind.rank == "vector"
    with pytest.raises(ValueError, match=r"a Velocity \(a vector\) cannot have values of 1 component"):
        td.Field(td.H1(grid), tack.zeros((n,)), "m/s", kind="Velocity")
    with pytest.raises(ValueError, match="3 components"):
        td.Field(td.H1(grid), vectors, "Pa", kind="Pressure")
    tensors = tack.zeros((n,), tack.mat3())
    assert td.Field(td.H1(grid), tensors, kind=td.Kind(rank="tensor")).kind.rank == "tensor"
    symmetric = tack.zeros((n,), tack.vec(6, tack.f32))      # XX YY ZZ XY YZ XZ
    assert td.Field(td.H1(grid), symmetric, "Pa", kind="Stress").kind.name == "Stress"
    flat = tack.zeros((n,), tack.vec(2, tack.f32))
    assert td.Field(td.H1(grid), flat, kind=td.Kind(rank="vector")).kind.rank == "vector"


def test_a_vector_space_holds_a_vector_whatever_its_width(grid):
    space = td.HDiv(grid)
    flux = td.Field(space, tack.zeros((space.size,)), "m/s", kind="Velocity")
    assert flux.kind.rank == "vector"
    with pytest.raises(ValueError, match="HDiv"):
        td.Field(space, tack.zeros((space.size,)), "Pa", kind="Pressure")


def test_a_vector_space_takes_only_its_own_variance(grid):
    hdiv, hcurl = td.HDiv(grid), td.HCurl(grid)
    flux = td.Field(hdiv, tack.zeros((hdiv.size,)), "T", kind="MagneticFluxDensity")
    assert flux.kind.variance is None
    with pytest.raises(ValueError, match="covariant, which a field in HDiv is not"):
        td.Field(hdiv, tack.zeros((hdiv.size,)), "V/m", kind="ElectricField")
    td.Field(hcurl, tack.zeros((hcurl.size,)), "V/m", kind="ElectricField")
    with pytest.raises(ValueError, match="contravariant, which a field in HCurl is not"):
        td.Field(hcurl, tack.zeros((hcurl.size,)), "m/s", kind="Velocity")


def test_moving_and_converting_keep_the_kind(grid):
    grid.fields["p"] = scalar(grid, "Pa")
    second = dispatch.as_device(_SecondCPU())
    try:
        assert grid.to_device(second).fields["p"].kind == td.Kind("Pressure")
    finally:
        tack.set_default_device("cpu")
    converted = td.convert(grid, "kPa", "p").fields["p"]
    assert converted.unit == "kPa" and converted.kind == td.Kind("Pressure")


def test_common_gives_one_kind_or_names_two():
    p = td.Kind("Pressure")
    assert kinds.common("the pieces' field 'p'", [p, td.Kind("Pressure")]) == p
    assert kinds.common("the pieces' field 'p'", [p, None]) is None
    assert kinds.common("the pieces' field 'p'", []) is None
    with pytest.raises(ValueError, match="the pieces' field 'p'"):
        kinds.common("the pieces' field 'p'", [p, td.Kind("Stress")])


def test_a_density_names_the_total_it_integrates_to():
    assert kinds.density_of("Density") == td.Kind("Mass")
    assert kinds.density_of(td.Kind("EnergyStagnationDensity")) == td.Kind("Energy")
    assert kinds.density_of("Temperature") is None
    assert kinds.density_of(None) is None


def test_the_registry_takes_new_kinds_once():
    kinds.add("TestTorque", "N*m", "vector", "intensive", parity="axial")
    try:
        assert td.Kind("TestTorque").variance == "contravariant"
        kinds.add("TestTorque", "N*m", "vector", "intensive", parity="axial")    # the same
        with pytest.raises(ValueError, match="registered already"):
            kinds.add("TestTorque", "J", "scalar", "extensive")
        assert "TestTorque" in kinds.names()
    finally:
        del kinds._REGISTRY["TestTorque"]
    with pytest.raises(KeyError, match="not registered"):
        kinds.add("TestDensity", "kg/m^3", "scalar", "intensive", density_of="TestMass")
    with pytest.raises(ValueError, match="not extensive"):
        kinds.add("TestDensity", "1/m^3", "scalar", "intensive", density_of="Temperature")
    assert "TestDensity" not in kinds.names()
    kinds.add("TestFlux", "W/m^2", "vector", "intensive", variance=None)
    try:
        assert td.Kind("TestFlux").variance is None
    finally:
        del kinds._REGISTRY["TestFlux"]


def test_the_registry_is_seeded_from_cgns_names():
    for name in ("Density", "Pressure", "Temperature", "Velocity", "Momentum", "Vorticity",
                 "Stress", "Strain", "Mach"):
        assert name in kinds.names()
    for name in kinds.names():
        k = td.Kind(name)
        assert k.rank is not None and k.extent is not None and k.parity is not None
        if k.rank == "scalar":
            assert k.variance is None
    for name in ("Momentum", "MagneticFluxDensity", "CurrentDensity", "HeatFlux"):
        assert td.Kind(name).variance is None          # flux densities turn as HDiv does
    assert td.Kind("Velocity").variance == "contravariant"


@pytest.mark.parametrize("call", [
    lambda d: td.extract_cells(d, [0, 1, 2]),
    lambda d: td.threshold(d, "p", -1.0, 1.0),
    lambda d: td.clip(d, "x", 0.5),
    lambda d: td.shrink(d),
    lambda d: td.project(d, "p", td.Constant(d)),
], ids=["extract_cells", "threshold", "clip", "shrink", "project"])
def test_filters_of_the_same_values_keep_the_kind(grid, call):
    grid.fields["p"] = scalar(grid, "Pa")
    grid.fields["x"] = td.Field(td.H1(grid), tack.array(grid.positions()[:, 0].copy()), "m")
    out = call(grid)
    p = out if isinstance(out, td.Field) else out.fields["p"]
    assert p.kind == td.Kind("Pressure") and p.unit == "Pa"


def test_a_temporal_step_keeps_the_kind(grid):
    from tack.dataset.temporal import _Step

    grid.fields["p"] = scalar(grid, "Pa")
    assert _Step(grid, ["p"]).fields["p"].kind == td.Kind("Pressure")


def test_carrying_leaves_a_kind_that_no_longer_fits_unknown(grid):
    from tack.dataset.carry import _with_kind

    faces = td.Values(grid, "faces")
    fluxes = td.Field(faces, tack.zeros((faces.size,)), "m/s")      # an HDiv vector's values
    assert _with_kind(fluxes, td.Kind("Velocity")).kind is None
    pressures = td.Field(faces, tack.zeros((faces.size,)), "Pa")
    assert _with_kind(pressures, td.Kind("Pressure")).kind == td.Kind("Pressure")
