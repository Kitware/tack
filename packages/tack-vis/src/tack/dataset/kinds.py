# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""What a field's values are: its kind (``Field.kind``), beside its unit.

A unit says what the numbers are in; a kind says what they are. Pressure and
stress share a unit, as do torque and energy, and a vector may be a velocity,
a vorticity (a pseudovector, which a reflection turns the other way) or three
unrelated numbers. A ``Kind`` records the parts that tell them apart, each
optional:

- ``name``: the quantity, from the registry (``add``, ``names``);
- ``rank``: ``"scalar"``, ``"vector"``, ``"tensor"`` or ``"symmetric tensor"``;
- ``extent``: ``"intensive"`` (a density, a velocity) or ``"extensive"`` (a
  mass, a count), by what the field evaluates to, not how it is stored (a
  first-order ``HDiv`` velocity stores fluxes and is intensive);
- ``parity``: ``"polar"`` or ``"axial"`` (a pseudovector or pseudoscalar);
- ``variance``: how a vector or tensor turns under a linear map,
  ``"contravariant"`` (a velocity), ``"covariant"`` (a gradient, a normal) or,
  for a tensor, ``"mixed"`` (a velocity gradient). A flux density (a magnetic
  flux density, a current density, a heat flux) turns as ``HDiv`` does, by the
  contravariant Piola map, which needs a density weight a kind does not record
  yet: its variance is left unknown.

Given a name, the registry fills the parts not given, and the parts given
must agree with it. Without one, each part stands alone: a field read from
VTK's active vectors would have a rank and nothing else. ``None`` is unknown, part
by part, as for units: nothing is guessed.

A kind never converts anything, and nothing reads one yet: filters keep
treating fields as they did, until each declares what it does to a kind
(``docs/design/plans/field-quantities.md``, D2).

The registry is seeded from CGNS's data names (``Pressure``, ``Velocity``,
``Stress``), with a few extensive totals they are densities of (``Mass``,
``Energy``), each with the unit its values have the dimensions of. A
symmetric tensor's six components are in VTK's order, XX YY ZZ XY YZ XZ.
"""

from dataclasses import dataclass, fields

from tack.dataset import units

__all__ = ["Kind", "add", "common", "density_of", "names"]

RANKS = ("scalar", "vector", "tensor", "symmetric tensor")
EXTENTS = ("intensive", "extensive")
PARITIES = ("polar", "axial")
VARIANCES = ("contravariant", "covariant", "mixed")


@dataclass(frozen=True)
class Kind:
    """What a field's values are; every part optional (``None``: unknown). Given a
    ``name`` from the registry, the parts not given are the registry's, and the
    parts given must agree with it. Kinds compare by value."""

    name: str = None
    rank: str = None
    extent: str = None
    parity: str = None
    variance: str = None

    def __post_init__(self):
        for part, allowed in (("rank", RANKS), ("extent", EXTENTS), ("parity", PARITIES),
                              ("variance", VARIANCES)):
            given = getattr(self, part)
            if given is not None and given not in allowed:
                raise ValueError(f"a kind's {part} is one of {', '.join(map(repr, allowed))}, "
                                 f"not {given!r}")
        if self.name is not None:
            self._fill_from_registry()
        if self.variance is not None and self.rank == "scalar":
            raise ValueError(f"{self._described()} has no variance: a scalar does not turn "
                             "under a map")
        if self.variance == "mixed" and self.rank == "vector":
            raise ValueError(f"{self._described()} is contravariant or covariant; only a "
                             "tensor is mixed")

    def _fill_from_registry(self):
        entry = _REGISTRY.get(self.name)
        if entry is None:
            raise KeyError(f"no kind {self.name!r} in the registry: add it first "
                           "(tack.dataset.kinds.add)")
        for part in _PARTS:
            given, registered = getattr(self, part), getattr(entry.kind, part)
            if given is None:
                object.__setattr__(self, part, registered)
            elif registered is not None and given != registered:
                raise ValueError(f"a {self.name} is {registered!r} ({part}), not {given!r}")

    def _described(self):
        """``"a Velocity (a vector)"``, ``"a vector"`` or ``"a kind"``, for messages."""
        rank = None if self.rank is None else f"a {self.rank}"
        if self.name is None:
            return rank or "a kind"
        return f"a {self.name}" + ("" if rank is None else f" ({rank})")

    @property
    def reference_unit(self):
        """A unit with the dimensions this kind's values have, from the registry, or
        ``None`` without a name."""
        entry = _REGISTRY.get(self.name)
        return None if entry is None else units.canonical(entry.unit)

    def __repr__(self):
        given = [f"{f.name}={getattr(self, f.name)!r}" for f in fields(self)
                 if getattr(self, f.name) is not None]
        return f"Kind({', '.join(given)})"


_PARTS = ("rank", "extent", "parity", "variance")


@dataclass(frozen=True)
class _Entry:
    kind: Kind
    unit: str
    density_of: str


_REGISTRY = {}


_BY_RANK = "by rank"


def add(name, unit, rank, extent=None, parity="polar", variance=_BY_RANK, density_of=None):
    """Register kind ``name``: its values have ``unit``'s dimensions, and its rank,
    extent, parity and variance are as given (by default a vector's variance is
    contravariant and a tensor's or a scalar's unknown; pass ``None`` to leave a
    vector's unknown). ``density_of`` names the extensive kind a density
    integrates to (``"Mass"`` for ``"Density"``). A name registered already must
    be given the same parts."""
    units.canonical(unit)                       # refuses what is not a unit
    return _register(name, unit, rank, extent, parity, variance, density_of)


def _register(name, unit, rank, extent=None, parity="polar", variance=_BY_RANK,
              density_of=None):
    """``add`` without parsing ``unit``, so seeding the registry imports no unyt."""
    if not isinstance(name, str) or not name:
        raise TypeError(f"a kind's name is a non-empty string, not {name!r}")
    if variance == _BY_RANK:
        variance = "contravariant" if rank == "vector" else None
    kind = Kind(rank=rank, extent=extent, parity=parity, variance=variance)
    object.__setattr__(kind, "name", name)      # registered below, so not looked up
    if density_of is not None:
        target = _REGISTRY.get(density_of)
        if target is None:
            raise KeyError(f"{name!r} is a density of {density_of!r}, which is not registered")
        if target.kind.extent != "extensive":
            raise ValueError(f"{name!r} is a density of {density_of!r}, which is not "
                             "extensive: a density integrates to a total")
    known = _REGISTRY.get(name)
    if known is not None and (known.kind != kind or known.density_of != density_of
                              or not units.same(units.canonical(known.unit),
                                                units.canonical(unit))):
        raise ValueError(f"kind {name!r} is registered already, as {known.kind!r} in "
                         f"{known.unit!r}")
    _REGISTRY[name] = _Entry(kind, unit, density_of)
    return kind


def names():
    """The registered kinds' names, sorted."""
    return sorted(_REGISTRY)


def density_of(kind):
    """The extensive kind ``kind`` (a ``Kind``, a name or ``None``) integrates to, or
    ``None``: a mass density to a mass; a temperature or a velocity to none."""
    name = kind.name if isinstance(kind, Kind) else kind
    entry = _REGISTRY.get(name)
    return None if entry is None or entry.density_of is None else Kind(entry.density_of)


def common(what, kinds):
    """The one kind of ``kinds`` (``Kind`` or ``None`` each), or ``ValueError`` naming
    ``what`` (``"the pieces' field 'p'"``, say) when two known kinds differ; ``None``
    if any is unknown, as ``units.common`` does for units."""
    kinds = list(kinds)
    known = [kind for kind in kinds if kind is not None]
    for kind in known[1:]:
        if kind != known[0]:
            raise ValueError(f"{what} is a {known[0]!r} and a {kind!r}")
    return known[0] if known and len(known) == len(kinds) else None


def coerce(kind):
    """``kind`` as a ``Kind``: a ``Kind`` as it is, a registered name as its kind,
    ``None`` as ``None``."""
    if kind is None or isinstance(kind, Kind):
        return kind
    if isinstance(kind, str):
        return Kind(kind)
    raise TypeError(f"a field's kind is a tack.dataset.Kind or a registered name, not {kind!r}")


def width_fits(rank, width, vector_space=False):
    """Whether values of ``width`` components (1 for a scalar) can be of ``rank``. A
    field in ``HDiv`` or ``HCurl`` (``vector_space``) is a vector whatever its width."""
    if rank is None:
        return True
    if vector_space:
        return rank == "vector"
    return width in {"scalar": (1,), "vector": (2, 3), "tensor": (4, 9),
                     "symmetric tensor": (3, 6)}[rank]


def _seed():
    """CGNS's data names (tack-expr's ``fx.cgns`` dictionary), with the totals some
    of them are densities of."""
    _register("Mass", "kg", "scalar", "extensive")
    _register("Energy", "J", "scalar", "extensive")
    _register("LinearMomentum", "kg*m/s", "vector", "extensive")
    for name, unit, extent, density in (
        ("Density", "kg/m^3", "intensive", "Mass"),
        ("Pressure", "Pa", "intensive", None),
        ("Temperature", "K", "intensive", None),
        ("EnergyInternal", "J/kg", "intensive", None),
        ("Enthalpy", "J/kg", "intensive", None),
        ("Entropy", "J/(kg*K)", "intensive", None),
        ("EnergyStagnationDensity", "J/m^3", "intensive", "Energy"),
        ("PressureStagnation", "Pa", "intensive", None),
        ("TemperatureStagnation", "K", "intensive", None),
        ("VelocitySound", "m/s", "intensive", None),
        ("VelocityMagnitude", "m/s", "intensive", None),
        ("Mach", "1", "intensive", None),
        ("ViscosityMolecular", "Pa*s", "intensive", None),
        ("ViscosityKinematic", "m^2/s", "intensive", None),
        ("ViscosityEddy", "Pa*s", "intensive", None),
        ("ThermalConductivity", "W/(m*K)", "intensive", None),
        ("TurbulentEnergyKinetic", "m^2/s^2", "intensive", None),
        ("TurbulentDissipation", "m^2/s^3", "intensive", None),
        ("TurbulentDissipationRate", "1/s", "intensive", None),
        ("TurbulentDistance", "m", "intensive", None),
        ("CoefPressure", "1", "intensive", None),
        ("VonMisesStress", "Pa", "intensive", None),
        ("YoungsModulus", "Pa", "intensive", None),
        ("PoissonRatio", "1", "intensive", None),
        ("EquivalentPlasticStrain", "1", "intensive", None),
        ("ElectricPotential", "V", "intensive", None),
        ("ElectricConductivity", "A/(V*m)", "intensive", None),
    ):
        _register(name, unit, "scalar", extent, "polar", None, density)
    for name, unit, parity, variance, density in (
        ("Velocity", "m/s", "polar", "contravariant", None),
        ("Momentum", "kg/(m^2*s)", "polar", None, "LinearMomentum"),
        ("Vorticity", "1/s", "axial", "covariant", None),
        ("Displacement", "m", "polar", "contravariant", None),
        ("Force", "N", "polar", "contravariant", None),
        ("ElectricField", "V/m", "polar", "covariant", None),
        ("MagneticField", "A/m", "axial", "covariant", None),
        ("MagneticFluxDensity", "T", "axial", None, None),
        ("CurrentDensity", "A/m^2", "polar", None, None),
        ("HeatFlux", "W/m^2", "polar", None, None),
    ):
        _register(name, unit, "vector", "intensive", parity, variance, density)
    _register("Stress", "Pa", "symmetric tensor", "intensive", variance="contravariant")
    _register("Strain", "1", "symmetric tensor", "intensive", variance="covariant")


_seed()
