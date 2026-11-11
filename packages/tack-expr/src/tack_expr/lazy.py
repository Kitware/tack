# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""An expression as a field evaluated where an algorithm reads it, never stored.

fx.field(text, data) is a td.Field whose view computes the expression at each point an
algorithm asks for: value(c, pc), corner_value(c, j), and value_range(c), a bound in each
cell by interval arithmetic on its fields' Bernstein bounds, for the algorithms that skip
cells by it (unbounded where the expression has no rule). Tack's algorithms take it as
they take a stored field, and fx.field(text) where a field goes
(td.contour(data, fx.field("mag(U)"), 1.0)), read on each dataset or piece;
a contour subdivides by its inputs' largest order. A filter's output has the expression
again, over the fields carried there, and writers store its values.
"""

import itertools
import re
import warnings

import tack
import tack.dataset as td
from tack.dataset.spaces import Space

from . import parser as P

_keys = itertools.count(1)


class ExpressionSpace(Space):
    """The space of an expression field: one notional value per cell, read through the
    generated view. continuous when the expression is pointwise in continuous fields, so
    contour shares its points between cells."""

    on = "cells"
    interpolated = True
    evaluated = True

    @classmethod
    def _params(cls, key=0):
        return {"key": key}

    @property
    def size(self):
        return self.topology.num_cells

    def mixin_for(self, key, shape=None):
        return self._mixin

    def rebind(self, field, data):
        """The expression on ``data``, a filter's output, over the fields carried there;
        None, with a warning, where one of them was not."""
        names = {}
        for bound, f in self.names.items():
            stored = next((n for n, g in self._data.fields.items() if g is f), None)
            if stored is None or stored not in data.fields:
                warnings.warn(f"{self.text!r} is not carried to the output: a field it reads is not there",
                              stacklevel=2)
                return None
            names[bound] = data.fields[stored]
        try:
            with P.bound({**(P._BINDINGS.get() or {}), **names}):
                f = _field(self.text, data)
        except (P.ExprError, KeyError, TypeError, ValueError) as error:
            warnings.warn(f"{self.text!r} is not carried to the output: {error}", stacklevel=2)
            return None
        f.space.names = {**f.space.names, **names}
        return td.Field(f.space, f.values, field.unit)

    def stored(self, field, data):
        """Its values for a writer: at the points where the expression is continuous,
        otherwise at the cell centers."""
        with P.bound({**(P._BINDINGS.get() or {}), **self.names}):
            out = P.evaluate(self.text, data, at="points" if self.continuous else "cells")
        return td.Field(out.space, out.values, field.unit)

    def attributes(self, group=None, dtype=None):
        if group is None:
            return {}
        data = self._data
        out = {"dom": data.domain_view("cells", group, incidence=self._incidence),
               "no_hessian": float("nan")}
        for prefix, args, consts in self._parts:
            memo = {}
            for k, (kind, get) in enumerate(args):
                payload = get(data, memo)
                out[f"{prefix}{k}"] = payload.view(group) if kind == "field" else payload
            for k, v in enumerate(consts):
                out[f"{prefix.lower()}c{k}"] = float(P._scalar(data, v))
        return out


def _method(lines, result, prefix, name, signature, head=()):
    swap = re.compile(r"\b(A\d+|c\d+|dom)\b")

    def sub(text):
        def rename(m):
            token = m.group(1)
            if token == "dom":
                return "self.dom"
            return f"self.{prefix}{token[1:]}" if token[0] == "A" else f"self.{prefix.lower()}{token}"
        return swap.sub(rename, text)

    body = list(head) + [sub(line) for line in lines] + [f"return {sub(result)}"]
    return f"\n    @tack.func\n    def {name}({signature}):\n" + "".join(f"        {b}\n" for b in body)


class Expression:
    """``field(text)``: the expression as a field argument for any of Tack's algorithms,
    which read it on each dataset they are given (``on``), each piece of a
    PartitionedDataSet included."""

    def __init__(self, text, fields=None, units=None):
        self.text, self.fields, self.units = text, fields, units

    def on(self, data):
        """The expression on ``data``, a td.Field (``field(text, data)``)."""
        return field(self.text, data, self.fields, self.units)

    def __repr__(self):
        return f"tack_expr.field({self.text!r})"


def field(text, data=None, fields=None, units=None):
    """The expression as a td.Field evaluated wherever it is read: by contour, clip,
    probe and the other algorithms that read a field through its view. fields and units:
    as tack_expr.evaluate's; the field's unit is the expression's. Without ``data``, an
    Expression, which an algorithm makes the field of on each dataset it reads it on."""
    if data is None:
        return Expression(text, fields, units)
    from . import _front
    if _front.needs_front(text, fields, units, data) or _front.expands(text, data):
        text, names, unit = _front.prepare(text, data, fields, units)
        with P.bound({**(P._BINDINGS.get() or {}), **names}):
            f = _field(text, data)
        f.space.names = {**f.space.names, **names}
        return td.Field(f.space, f.values, unit)
    return _field(text, data)


def _field(text, data):
    c = P._Compiler(text, data, "cells")
    result, rtype = c.parse()
    if rtype != P.S:
        raise P.ExprError("a field to contour or clip is a scalar expression", 0, text)
    parts = [("A", list(c.args), list(c.consts))]
    methods = _method(P._live(c.lines, [result]), result, "A", "value", "self, c, pc")
    imports = P._imports(c)
    continuous = False
    try:
        k = P._Compiler(text, data, "corners")
        corner, _ = k.parse()
        parts.append(("B", list(k.args), list(k.consts)))
        head = ["pc = self.dom.parametric_point(j)"] if k.uses_pc else []
        methods += _method(P._live(k.lines, [corner]), corner, "B", "corner_value", "self, c, j", head)
        imports = "\n".join(sorted(set(imports.splitlines()) | set(P._imports(k).splitlines()))) + "\n"
        continuous = not c.uses_derivatives and not c.uses_world_gradient
    except (P.ExprError, P._NeedCorners):
        methods += ("\n    @tack.func\n    def corner_value(self, c, j):\n"
                    "        return self.value(c, self.dom.parametric_point(j))\n")
    ranges = _ranges(text, data)
    if ranges is None:
        methods += ("\n    @tack.func\n    def value_range(self, c):\n"
                    "        return -3.0e38, 3.0e38\n")
    else:
        methods += ranges[0]
        parts.append(("R", ranges[1], []))
    more, compilers = _derivative_methods(text, data, parts)
    methods += more
    for compiler in compilers:
        imports = "\n".join(sorted(set(imports.splitlines()) | set(P._imports(compiler).splitlines()))) + "\n"
    # Exact crossings read stored fields' entity values through dof; an expression has none
    # and is read through value there, but the method has to exist for the kernel to build.
    methods += ("\n    @tack.func\n    def dof(self, c, k):\n"
                "        return 0.0 * self.dom.parametric_center()[0]\n")
    source = imports + "\n\nclass Mixin:\n" + methods
    space = ExpressionSpace(data, key=next(_keys))
    space._mixin = P._load(source, "Mixin")
    space._incidence = P._incidence(source)
    space._data = data
    space._parts = parts
    space.continuous = continuous
    orders = [getattr(c.lookup(n, 0).space, "order", 1) for n in sorted(c.fields_read)]
    space.default_subdivision = max([1] + [o for o in orders if isinstance(o, int)])
    # An expression's polynomial order is not known in general: twice its inputs'
    # largest, exact for a product of two of them, a quadrature approximation otherwise.
    space.integration_order = 2 * space.default_subdivision
    space.source = source
    space.text = text
    # The bound fields it reads, so a later expression or a filter's output can read it again.
    space.names = {n: c.bindings[n] for n in c.fields_read if n in c.bindings}
    dtype = tack.f64 if data.dtype is tack.f64 else tack.f32
    return td.Field(space, tack.zeros((data.num_cells,), dtype, device=data.device))


def _derivative_methods(text, data, parts):
    """An expression field's world_gradient, and world_jet with its Hessian, compiled from
    grad() of the expression, so Tack's filters and kernels differentiate it as they do a
    stored field; none where the expression's fields have no derivatives."""
    try:
        g = P._Compiler(f"grad({text})", data, "cells")
        gradient, _ = g.parse()
    except (P.ExprError, P._NeedCorners):
        return "", []
    parts.append(("G", list(g.args), list(g.consts)))
    out = _method(P._live(g.lines, [gradient]), gradient, "G", "world_gradient", "self, cells, c, pc")
    rows = ", ".join(f"grad(grad({text})[{k}])" for k in range(3))
    try:
        h = P._Compiler("{" + rows + "}", data, "cells")
        hessian, _ = h.parse()
    except (P.ExprError, P._NeedCorners):
        return out, [g]
    parts.append(("H", list(h.args), list(h.consts)))
    out += _method(P._live(h.lines, [hessian]), hessian, "H", "expression_hessian", "self, cells, c, pc")
    out += ("\n    @tack.func\n    def world_jet(self, cells, c, pc):\n"
            "        return self.value(c, pc), self.world_gradient(cells, c, pc), "
            "self.expression_hessian(cells, c, pc)\n")
    return out, [g, h]


def _fields(text, data):
    c = P._Compiler(text, data, "cells")
    c.parse()
    return [c.lookup(n, 0) for n in sorted(c.fields_read)]


# --- a bound in each cell ----------------------------------------------------------

_MONOTONE = {"exp": "exp", "tanh": "tanh", "atan": "atan", "sinh": "sinh"}


class _Ranges:
    """Interval arithmetic on the expression: each cell's bound from its fields' Bernstein
    bounds (value_range), so a subdivided contour skips the cells it cannot cross. Anything
    without a rule (a derivative, a vector, a division by a field) is unbounded, which only
    keeps a cell."""

    def __init__(self, data):
        self.data = data
        self.bindings = P._BINDINGS.get() or {}
        self.lines, self.args, self.keys = [], [], {}

    def tmp(self, lo, hi):
        k = len(self.lines)
        self.lines.append(f"rl{k} = {lo}")
        self.lines.append(f"rh{k} = {hi}")
        return f"rl{k}", f"rh{k}"

    def field(self, name):
        field = self.bindings.get(name) or (P._named_field(self.data, name, "cells")
                                            if name in self.data.fields or P._derived(name) else None)
        if field is None or P._type(field) != P.S:
            return None
        kind = P._kind(field)
        if kind not in ("basis", "values-cells"):
            return None
        if name not in self.keys:
            self.keys[name] = len(self.args)
            getter = (lambda d, m, f=field: f) if name in self.bindings else (lambda d, m, n=name: P._named_field(d, n, "cells"))
            self.args.append(("field", getter))
            k = len(self.lines)
            self.lines.append(f"rl{k}, rh{k} = self.R{self.keys[name]}.value_range(c)")
            self.keys[(name, "range")] = (f"rl{k}", f"rh{k}")
        return self.keys[(name, "range")]

    def range(self, node):
        """(lo, hi) codes, or None for unbounded."""
        from . import ops as O
        if isinstance(node, O._Const):
            v = repr(float(node.value))
            return v, v
        if isinstance(node, O._Ref):
            return self.field(node.name)
        if isinstance(node, O._Name) and node.name == "pi":
            return "3.141592653589793", "3.141592653589793"
        if isinstance(node, O._Unary):
            r = self.range(node.a)
            return None if r is None else self.tmp(f"-{r[1]}", f"-{r[0]}")
        if isinstance(node, O._Binary):
            return self.binary(node)
        if isinstance(node, O._Call):
            return self.call(node)
        return None

    def binary(self, node):
        op = node.op
        if op in ("<", "<=", ">", ">=", "==", "!=", "and", "or"):
            return "0.0", "1.0"
        a, b = self.range(node.a), self.range(node.b)
        if a is None or b is None:
            return None
        (al, ah), (bl, bh) = a, b
        if op == "+":
            return self.tmp(f"{al} + {bl}", f"{ah} + {bh}")
        if op == "-":
            return self.tmp(f"{al} - {bh}", f"{ah} - {bl}")
        if op == "*":
            p = [f"{x} * {y}" for x in (al, ah) for y in (bl, bh)]
            return self.tmp(f"min({p[0]}, {p[1]}, {p[2]}, {p[3]})", f"max({p[0]}, {p[1]}, {p[2]}, {p[3]})")
        from . import ops as O
        if op == "/" and isinstance(node.b, O._Const) and node.b.value != 0.0:
            s = repr(1.0 / float(node.b.value))
            return self.tmp(f"min({al} * {s}, {ah} * {s})", f"max({al} * {s}, {ah} * {s})")
        if op == "^" and isinstance(node.b, O._Const) and float(node.b.value).is_integer() and node.b.value > 0:
            n = int(node.b.value)
            if n % 2:
                return self.tmp(f"{al} ** {n}.0", f"{ah} ** {n}.0")
            lo, hi = self.absolute(al, ah)
            return self.tmp(f"{lo} ** {n}.0", f"{hi} ** {n}.0")
        return None

    def absolute(self, lo, hi):
        return self.tmp(f"tack.select({lo} >= 0.0, {lo}, tack.select({hi} <= 0.0, -{hi}, 0.0))",
                        f"max(-{lo}, {hi})")

    def call(self, node):
        fn, args = node.name, node.args
        if fn in ("sin", "cos"):
            return "-1.0", "1.0"
        ranges = [self.range(a) for a in args]
        if any(r is None for r in ranges) or not ranges:
            return None
        (lo, hi) = ranges[0]
        if fn in _MONOTONE and len(ranges) == 1:
            f = _MONOTONE[fn]
            return self.tmp(f"{f}({lo})", f"{f}({hi})")
        if fn == "sqrt":
            return self.tmp(f"sqrt(max({lo}, 0.0))", f"sqrt(max({hi}, 0.0))")
        if fn in ("ln", "log"):
            return self.tmp(f"tack.select({lo} > 0.0, log(max({lo}, 1.0e-30)), -3.0e38)",
                            f"tack.select({hi} > 0.0, log(max({hi}, 1.0e-30)), -3.0e38)")
        if fn == "abs":
            return self.absolute(lo, hi)
        if fn in ("min", "max") and len(ranges) == 2:
            (bl, bh) = ranges[1]
            return self.tmp(f"{fn}({lo}, {bl})", f"{fn}({hi}, {bh})")
        if fn == "clamp" and len(ranges) == 3:
            (xl, xh), (cl, ch) = ranges[1], ranges[2]
            return self.tmp(f"min(max({xl}, {lo}), {cl})", f"min(max({xh}, {hi}), {ch})")
        if fn == "if" and len(ranges) == 3:
            (al, ah), (bl, bh) = ranges[1], ranges[2]
            return self.tmp(f"min({al}, {bl})", f"max({ah}, {bh})")
        return None


def _ranges(text, data):
    """(the value_range method's source, its field arguments) for the expression, or None."""
    from . import _front
    try:
        tree = _front.parse(text)
    except P.ExprError:
        return None
    r = _Ranges(data)
    bound = r.range(tree)
    if bound is None:
        return None
    lo, hi = bound
    body = [*r.lines, f"return {lo}, {hi}"]
    return "\n    @tack.func\n    def value_range(self, c):\n" + "".join(f"        {b}\n" for b in body), r.args
