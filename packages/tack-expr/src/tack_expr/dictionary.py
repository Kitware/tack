# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A field dictionary: the names expressions use, mapped to the arrays a dataset stores.

Each entry is a quantity with a name, aliases and the unit its values are stored in. Bound to
a dataset, an entry finds its storage: a field of its name or an alias (exactly, then ignoring
case), a source field or expression given for that dataset, three component fields
(VelocityX, VelocityY, VelocityZ) as a vector, or a real and an imaginary part as a complex
field. So one expression, ``Pressure - 101.325[kPa]``, runs on a solver's output that calls
it ``p`` in psi and on another's that calls it ``PRESSURE`` in Pa.

cgns() gives the CGNS SIDS data names for flow quantities, with names in the same style for
structural ones, in SI.
"""

from dataclasses import dataclass, field, replace

from tack.dataset.units import canonical

_COMPLEX_SUFFIXES = (("_real", "_imag"), ("_re", "_im"), ("Real", "Imag"), ("_r", "_i"))


@dataclass(frozen=True)
class Entry:
    name: str
    unit: str | None = None
    aliases: tuple = ()
    source: str | None = None
    components: tuple | None = None
    complex: tuple | None = None
    description: str = ""


@dataclass
class Binding:
    """What a name stands for on one dataset: kind 'field' (the stored field's name), 'expr'
    (an expression over stored fields) or 'complex' (two Bindings, real and imaginary)."""
    kind: str
    target: object
    unit: object = None
    entry: Entry | None = None


@dataclass
class FieldDictionary:
    """Entries by name. length: the geometry's unit where the dataset gives none (DataSet.unit)."""
    entries: dict = field(default_factory=dict)
    length: str | None = None

    def define(self, name, unit=None, aliases=(), source=None, components=None, complex=None,
               description=""):
        """Add or replace the entry for name; returns the dictionary. unit: the unit values are
        stored in, for fields with none of their own (Field.unit)."""
        self.entries[name] = Entry(name, canonical(unit), tuple(aliases), source,
                                   tuple(components) if components else None,
                                   tuple(complex) if complex else None, description)
        return self

    def with_units(self, length=None, **units):
        """A copy whose entries are stored in these units: with_units(Pressure="psi", length="mm")."""
        copy = FieldDictionary(dict(self.entries), length or self.length)
        for name, unit in units.items():
            entry = copy.entries.get(name) or copy.entries.get(self._canonical(name))
            if entry is None:
                raise KeyError(f"no entry '{name}' in the dictionary")
            copy.entries[entry.name] = replace(entry, unit=canonical(unit))
        return copy

    def with_sources(self, **sources):
        """A copy reading these entries from the given field names or expressions."""
        copy = FieldDictionary(dict(self.entries), self.length)
        for name, source in sources.items():
            entry = copy.entries.get(name) or copy.entries.get(self._canonical(name))
            if entry is None:
                copy.define(name, source=source)
            else:
                copy.entries[entry.name] = replace(entry, source=source)
        return copy

    def _canonical(self, name):
        for entry in self.entries.values():
            if name in entry.aliases:
                return entry.name
        return None

    def names(self, entry):
        return (entry.name, *entry.aliases)

    def bind(self, data):
        """{name: Binding} for every name and alias this dictionary resolves on data, and for the
        stored fields the entries found (so a stored name carries its unit too)."""
        stored = set(data.fields.keys()) if data is not None else set()
        lower = {}
        for name in sorted(stored):
            lower.setdefault(name.lower(), name)

        def exact(name):
            return name if name in stored else None

        def ignoring_case(name):
            # Short names are too often something else in another case: u is not U.
            return exact(name) or (lower.get(name.lower()) if len(name) > 3 else None)

        out = {}
        later = []
        for entry in self.entries.values():
            binding = self._bind(entry, exact) or self._bind(entry, ignoring_case)
            if binding is None:
                later.append(entry)
                continue
            self._add(out, entry, binding)
        for entry in later:   # components that are entries themselves, complex ones among them
            if entry.components and all(p in out for p in entry.components):
                self._add(out, entry, Binding("expr", _assembled(list(entry.components)), entry.unit, entry))
        return out

    def _add(self, out, entry, binding):
        for name in self.names(entry):
            out.setdefault(name, binding)
        if binding.kind == "field":
            out.setdefault(binding.target, binding)
        elif binding.kind == "complex":
            for part in binding.target:
                if part.kind == "field":
                    out.setdefault(part.target, part)

    def _bind(self, entry, find):
        unit = entry.unit
        if entry.source is not None:
            found = find(entry.source)
            if found is not None:
                return Binding("field", found, unit, entry)
            if entry.source != entry.name:
                return Binding("expr", entry.source, unit, entry)
        if entry.complex is not None:
            parts = [find(p) for p in entry.complex]
            if all(parts):
                return Binding("complex", tuple(Binding("field", p, unit, entry) for p in parts), unit, entry)
            return None
        for name in self.names(entry):
            found = find(name)
            if found is not None:
                return Binding("field", found, unit, entry)
        for name in self.names(entry):
            for re_suffix, im_suffix in _COMPLEX_SUFFIXES:
                re_name, im_name = find(name + re_suffix), find(name + im_suffix)
                if re_name and im_name:
                    return Binding("complex", (Binding("field", re_name, unit, entry),
                                               Binding("field", im_name, unit, entry)), unit, entry)
        if entry.components is not None:
            parts = [find(p) for p in entry.components]
            if all(parts):
                return Binding("expr", _assembled(parts), unit, entry)
        return None


def _quoted(name):
    return name if name.isidentifier() else f'"{name}"'


def _assembled(parts):
    """A vector of three components, or a symmetric tensor of six in VTK's order (XX YY ZZ XY YZ XZ)."""
    q = [_quoted(p) for p in parts]
    if len(q) == 6:
        q = [f"{{{q[0]}, {q[3]}, {q[5]}}}", f"{{{q[3]}, {q[1]}, {q[4]}}}", f"{{{q[5]}, {q[4]}, {q[2]}}}"]
    return "{" + ", ".join(q) + "}"


def as_dictionary(fields):
    """A FieldDictionary from one, or from {name: unit} for the fields of those names."""
    if fields is None or isinstance(fields, FieldDictionary):
        return fields
    d = FieldDictionary()
    for name, unit in dict(fields).items():
        d.define(name, unit=unit, source=name)
    return d


def _vector(d, name, unit, aliases=(), description=""):
    xyz = tuple(f"{name}{c}" for c in "XYZ")
    d.define(name, unit, aliases, components=xyz, description=description)
    for part in xyz:
        d.define(part, unit)


def _tensor(d, name, unit, aliases=(), description=""):
    parts = tuple(f"{name}{c}" for c in ("XX", "YY", "ZZ", "XY", "YZ", "XZ"))
    d.define(name, unit, aliases, components=parts, description=description)
    for part in parts:
        d.define(part, unit)


def cgns(length=None):
    """CGNS SIDS data names (flow) and the same style for structural and electromagnetic
    quantities, in SI where a field has no unit of its own. Aliases are only names that rarely
    mean anything else: one-letter names such as u or E are left to a dictionary of your own."""
    d = FieldDictionary(length=length)
    for name, unit, aliases in (
        ("Density", "kg/m^3", ("rho", "density")),
        ("Pressure", "Pa", ("p", "pressure")),
        ("Temperature", "K", ("T", "temperature")),
        ("EnergyInternal", "J/kg", ()),
        ("Enthalpy", "J/kg", ()),
        ("Entropy", "J/(kg*K)", ()),
        ("EnergyStagnationDensity", "J/m^3", ("rhoE",)),
        ("PressureStagnation", "Pa", ()),
        ("TemperatureStagnation", "K", ()),
        ("VelocitySound", "m/s", ()),
        ("VelocityMagnitude", "m/s", ()),
        ("Mach", "1", ("mach",)),
        ("ViscosityMolecular", "Pa*s", ("mu",)),
        ("ViscosityKinematic", "m^2/s", ("nu",)),
        ("ViscosityEddy", "Pa*s", ("mut",)),
        ("ThermalConductivity", "W/(m*K)", ()),
        ("TurbulentEnergyKinetic", "m^2/s^2", ("tke",)),
        ("TurbulentDissipation", "m^2/s^3", ("epsilon",)),
        ("TurbulentDissipationRate", "1/s", ("omega",)),
        ("TurbulentDistance", "m", ()),
        ("CoefPressure", "1", ("Cp",)),
        ("VonMisesStress", "Pa", ("vonMises",)),
        ("YoungsModulus", "Pa", ()),
        ("PoissonRatio", "1", ()),
        ("EquivalentPlasticStrain", "1", ("PEEQ",)),
        ("ElectricPotential", "V", ()),
        ("ElectricConductivity", "A/(V*m)", ()),
    ):
        d.define(name, unit, aliases)
    _vector(d, "Velocity", "m/s", ("U", "velocity"))
    _vector(d, "Momentum", "kg/(m^2*s)", ("momentum",))
    _vector(d, "Vorticity", "1/s", ("vorticity",))
    _vector(d, "Displacement", "m", ("displacement",))
    _vector(d, "Force", "N", ("force",))
    _vector(d, "ElectricField", "V/m", ())
    _vector(d, "MagneticField", "A/m", ())
    _vector(d, "MagneticFluxDensity", "T", ())
    _vector(d, "CurrentDensity", "A/m^2", ())
    _vector(d, "HeatFlux", "W/m^2", ())
    _tensor(d, "Stress", "Pa", ("sigma", "stress"))
    _tensor(d, "Strain", "1", ("strain",))
    return d
