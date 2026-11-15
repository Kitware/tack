# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The expression as a tree before code generation: names bound through a field dictionary,
complex values split into real and imaginary parts, and units checked and converted.

Text and the operator expressions of ops share the tree (ops' Expr nodes). prepare() runs
the passes and renders calculator text over real fields, which the parser compiles as any
other. An expression that needs none of them is left as it is, so its kernel is the one it
always was.

Units are Tack's (Field.unit, DataSet.unit for the geometry, tack.dataset.units), with its
rules: an unknown unit is neither guessed nor refused, and what is derived from one is
unknown; known units that differ are converted where they meet (a sum, a comparison, a
choice) and refused when their dimensions differ; a bare number in a sum or comparison is in
the unit of what it meets; a derivative of a unit with an offset zero (degC) is in its delta
(K). Nothing is converted to SI that need not be: kPa + kPa stays kPa.
"""

import re

from tack.dataset import units as U

from . import ops as O
from . import parser as P
from .dictionary import _COMPLEX_SUFFIXES, as_dictionary

_NEEDS = re.compile(r"(?:\d|\.)\s*\[|\d[ij]\b|\b(?:real|imag|conj|phase|complex|arg)\s*\(")
_SYNTAX_NAMES = {"coords", "coordsX", "coordsY", "coordsZ", "iHat", "jHat", "kHat", "pi", "true", "false",
                 "identity"}
_OPS = {"&&": "and", "&": "and", "||": "or", "|": "or", "=": "==", "<>": "!="}


class _Quantity(O.Expr):
    """A number with a unit, 101.325[kPa]."""

    def __init__(self, value, unit):
        self.value, self.unit = value, unit

    def _leaves(self, out):
        pass


class _Complex(O.Expr):
    """A complex value from its real and imaginary parts."""

    def __init__(self, re, im):
        self.re, self.im = O._wrap(re), O._wrap(im)

    def _leaves(self, out):
        self.re._leaves(out)
        self.im._leaves(out)


class _Declared(O.Expr):
    """A subtree whose values are in a declared unit: a dictionary entry defined by an expression."""

    def __init__(self, a, unit):
        self.a, self.unit = a, unit

    def _leaves(self, out):
        self.a._leaves(out)

    def _text(self, names):
        return self.a._text(names)


def _at(node, pos):
    node.pos = pos
    return node


# --- text to tree -----------------------------------------------------------

class _Reader:
    def __init__(self, text):
        self.text = text
        self.toks = P._Compiler._tokenize(None, text)
        self.k = 0

    def error(self, message, pos):
        raise P.ExprError(message, pos, self.text)

    def failure(self, message, pos):
        return P.ExprError(message, pos, self.text)

    def peek(self):
        return self.toks[self.k]

    def take(self):
        self.k += 1
        return self.toks[self.k - 1]

    def expect(self, val):
        _, v, pos = self.take()
        if v != val:
            self.error(f"expected '{val}'", pos)

    def parse(self):
        tree = self.expr(0)
        kind, val, pos = self.peek()
        if kind != "end":
            self.error(f"unexpected '{val}'", pos)
        return tree

    def expr(self, rbp):
        left = self.prefix(self.take())
        while True:
            kind, val, _ = self.peek()
            bp = P._INFIX[val][0] if kind == "op" and val in P._INFIX else 0
            if rbp >= bp:
                return left
            left = self.infix(self.take(), left)

    def args(self, close):
        parts = []
        if self.peek()[1] != close:
            parts.append(self.expr(0))
            while self.peek()[1] == ",":
                self.take()
                parts.append(self.expr(0))
        self.expect(close)
        return parts

    def prefix(self, tok):
        kind, val, pos = tok
        if kind == "num":
            nkind, nval, npos = self.peek()
            if nkind == "name" and nval in ("i", "j") and npos == pos + len(val):
                self.take()
                return _at(_Complex(0.0, float(val)), pos)
            if nval == "[":
                self.take()
                start = npos + 1
                while self.peek()[1] != "]":
                    if self.peek()[0] == "end":
                        self.error("a unit in brackets needs its ']'", npos)
                    self.take()
                end = self.take()[2]
                try:
                    unit = U.canonical(self.text[start:end].strip())
                except (ValueError, TypeError) as error:
                    self.error(str(error), start)
                return _at(_Quantity(float(val), unit), pos)
            return _at(O._Const(float(val)), pos)
        if kind == "str":
            return _at(O._Ref(val[1:-1]), pos)
        if kind == "name":
            if self.peek()[1] == "(":
                self.take()
                return _at(O._Call(val, *self.args(")")), pos)
            return _at(O._Name(val) if val in _SYNTAX_NAMES else O._Ref(val), pos)
        if val == "(":
            tree = self.expr(0)
            self.expect(")")
            return tree
        if val == "{":
            return _at(O._Vector(*self.args("}")), pos)
        if val in ("-", "+", "!", "not"):
            a = self.expr(P._PREFIX_BP)
            if val == "+":
                return a
            return _at(O._Unary("-", a) if val == "-" else O._Call("not", a), pos)
        raise self.failure("the expression ends too early" if kind == "end" else f"unexpected '{val}'", pos)

    def infix(self, tok, left):
        _, op, pos = tok
        bp, right = P._INFIX[op]
        if op == "?":
            a = self.expr(0)
            self.expect(":")
            return _at(O._Call("if", left, a, self.expr(bp - 1)), pos)
        if op == "[":
            index = self.expr(0)
            self.expect("]")
            if not isinstance(index, O._Const) or index.value not in (0.0, 1.0, 2.0):
                self.error("an index must be 0, 1 or 2", pos)
            return _at(O._Index(left, int(index.value)), pos)
        b = self.expr(bp - 1 if right else bp)
        return _at(O._Binary(_OPS.get(op, op), left, b), pos)


def parse(text):
    return _Reader(text).parse()


# --- tree helpers -------------------------------------------------------------

def _children(node):
    if isinstance(node, O._Binary):
        return [node.a, node.b]
    if isinstance(node, (O._Unary, O._Index, _Declared)):
        return [node.a]
    if isinstance(node, O._Call):
        return list(node.args)
    if isinstance(node, O._Vector):
        return list(node.parts)
    if isinstance(node, _Complex):
        return [node.re, node.im]
    return []


def _like(node, *children):
    """A node as node, with these children."""
    if isinstance(node, O._Binary):
        new = O._Binary(node.op, *children)
    elif isinstance(node, O._Unary):
        new = O._Unary(node.op, *children)
    elif isinstance(node, O._Index):
        new = O._Index(children[0], node.k)
    elif isinstance(node, O._Call):
        new = O._Call(node.name, *children)
    elif isinstance(node, O._Vector):
        new = O._Vector(*children)
    elif isinstance(node, _Complex):
        new = _Complex(*children)
    elif isinstance(node, _Declared):
        new = _Declared(children[0], node.unit)
    else:
        return node
    if hasattr(node, "pos"):
        new.pos = node.pos
    return new


def _const(node):
    """The number a subtree of numbers is, or None."""
    if isinstance(node, O._Const):
        return node.value
    if isinstance(node, O._Unary):
        v = _const(node.a)
        return None if v is None else -v
    if isinstance(node, O._Binary) and node.op in "+-*/":
        a, b = _const(node.a), _const(node.b)
        if a is None or b is None or (node.op == "/" and b == 0.0):
            return None
        return {"+": a + b, "-": a - b, "*": a * b, "/": a / b if b else 0.0}[node.op]
    return None


def _has(node, kind):
    return isinstance(node, kind) or any(_has(c, kind) for c in _children(node))


def _calls(node):
    found = {node.name} if isinstance(node, O._Call) else set()
    for c in _children(node):
        found |= _calls(c)
    return found


class _Pass:
    def __init__(self, text):
        self.text = text

    def error(self, message, node):
        raise self.failure(message, node)

    def failure(self, message, node):
        return P.ExprError(message, getattr(node, "pos", 0) or 0, self.text)


# --- names through the dictionary ---------------------------------------------

_COMPONENTS = {"X": 0, "Y": 1, "Z": 2}


class _Binder(_Pass):
    def __init__(self, text, data, bindings):
        super().__init__(text)
        self.data, self.bindings = data, bindings
        self.stored = set(data.fields.keys()) if data is not None else set()
        self.derived = set(P._MESH) if data is not None else set()
        self.depth = 0

    def unit_of_stored(self, name, declared=None):
        """The field's own unit, else the one the dictionary declares for it."""
        field = (P._named_field(self.data, name) if name in self.stored or name in self.derived
                 else None)
        if getattr(field, "unit", None) is not None:
            return field.unit
        b = self.bindings.get(name)
        if declared is None and b is not None and b.kind == "field" and b.target == name:
            declared = b.unit
        return declared

    def ref(self, name, node):
        if name in self.stored:
            return _at(_with_unit(O._Ref(name), self.unit_of_stored(name)), getattr(node, "pos", 0))
        b = self.bindings.get(name)
        if b is None:
            for re_suffix, im_suffix in _COMPLEX_SUFFIXES:
                if name + re_suffix in self.stored and name + im_suffix in self.stored:
                    parts = (_with_unit(O._Ref(name + s), self.unit_of_stored(name + s)) for s in (re_suffix, im_suffix))
                    return _at(_Complex(*parts), getattr(node, "pos", 0))
            base, _, comp = name.rpartition("_")
            known = base and (base in self.stored or base in self.bindings)
            if known and comp in _COMPONENTS:
                return _at(O._Index(self.ref(base, node), _COMPONENTS[comp]), getattr(node, "pos", 0))
            if known and len(comp) == 2 and set(comp) <= set(_COMPONENTS):
                row = O._Index(self.ref(base, node), _COMPONENTS[comp[0]])
                return _at(O._Index(row, _COMPONENTS[comp[1]]), getattr(node, "pos", 0))
            return node
        return self.binding(b, node)

    def binding(self, b, node):
        pos = getattr(node, "pos", 0)
        if b.kind == "field":
            return _at(_with_unit(O._Ref(b.target), self.unit_of_stored(b.target, b.unit)), pos)
        if b.kind == "complex":
            re, im = (self.binding(part, node) for part in b.target)
            return _at(_Complex(re, im), pos)
        self.depth += 1
        if self.depth > 20:
            self.error(f"the dictionary's definition of '{b.entry.name}' refers to itself", node)
        try:
            inner = self.bind(parse(b.target))
        except P.ExprError as error:
            self.error(f"the dictionary's definition of '{b.entry.name}', {b.target!r}: {error.args[0]}", node)
        self.depth -= 1
        if b.unit is None or _known_units(inner):
            return _at(inner, pos) if not hasattr(inner, "pos") else inner
        return _at(_Declared(inner, b.unit), pos)

    def bind(self, node):
        if isinstance(node, O._Ref):
            return self.ref(node.name, node)
        if isinstance(node, O._Field):
            unit = getattr(node.field, "unit", None)
            if unit is None and self.data is not None:   # a stored field: the dictionary's unit for its name
                stored = [n for n, f in self.data.fields.entries() if f is node.field]
                unit = self.unit_of_stored(stored[0]) if len(stored) == 1 else None
            if unit is not None and getattr(node, "unit", None) is None:
                node.unit = unit
            return node
        if isinstance(node, O._Call) and node.name in P._FIELD_FUNCTIONS and len(node.args) == 1:
            arg = node.args[0]
            if isinstance(arg, O._Ref) and arg.name in self.stored:
                return _like(node, self.ref(arg.name, arg))
        return _like(node, *(self.bind(c) for c in _children(node)))


def _known_units(node):
    """Whether any leaf of the tree has a known unit, or is a quantity."""
    if getattr(node, "unit", None) is not None:
        return True
    return any(_known_units(c) for c in _children(node))


def _with_unit(node, unit):
    if unit is not None:
        node.unit = unit
    return node


# --- complex values -------------------------------------------------------------

_LINEAR = {"grad", "div", "curl", "laplacian", "points", "cells", "owner", "neighbor", "fsum", "trace",
           "transpose", "sym", "skew", "dev"}
_BILINEAR = {"dot", "cross", "outer", "ddot"}


def _zero_like(node):
    return O._Binary("*", O._Const(0.0), node)


class _Lowering(_Pass):
    def lower(self, node):
        """(real part, imaginary part or None)."""
        if isinstance(node, _Complex):
            ar, ai = self.lower(node.re)
            br, bi = self.lower(node.im)
            if ai is not None or bi is not None:
                self.error("complex() takes real parts", node)
            return ar, br
        if isinstance(node, O._Unary):
            r, i = self.lower(node.a)
            return _like(node, r), (None if i is None else O._Unary("-", i))
        if isinstance(node, O._Index):
            r, i = self.lower(node.a)
            return _like(node, r), (None if i is None else O._Index(i, node.k))
        if isinstance(node, _Declared):
            r, i = self.lower(node.a)
            return _Declared(r, node.unit), (None if i is None else _Declared(i, node.unit))
        if isinstance(node, O._Vector):
            parts = [self.lower(p) for p in node.parts]
            if all(i is None for _, i in parts):
                return _like(node, *(r for r, _ in parts)), None
            return (O._Vector(*(r for r, _ in parts)),
                    O._Vector(*(i if i is not None else _zero_like(r) for r, i in parts)))
        if isinstance(node, O._Binary):
            return self.binary(node)
        if isinstance(node, O._Call):
            return self.call(node)
        return node, None

    def binary(self, node):
        op = node.op
        (ar, ai), (br, bi) = self.lower(node.a), self.lower(node.b)
        if ai is None and bi is None:
            return _like(node, ar, br), None

        def B(o, x, y):
            return O._Binary(o, x, y)

        if op in ("+", "-"):
            if ai is None:
                return B(op, ar, br), (bi if op == "+" else O._Unary("-", bi))
            return B(op, ar, br), (ai if bi is None else B(op, ai, bi))
        if op == "*":
            if bi is None:
                return B("*", ar, br), B("*", ai, br)
            if ai is None:
                return B("*", ar, br), B("*", ar, bi)
            return B("-", B("*", ar, br), B("*", ai, bi)), B("+", B("*", ar, bi), B("*", ai, br))
        if op == "/":
            if bi is None:
                return B("/", ar, br), B("/", ai, br)
            den = B("+", B("*", br, br), B("*", bi, bi))
            if ai is None:
                return B("/", B("*", ar, br), den), B("/", O._Unary("-", B("*", ar, bi)), den)
            return (B("/", B("+", B("*", ar, br), B("*", ai, bi)), den),
                    B("/", B("-", B("*", ai, br), B("*", ar, bi)), den))
        raise self.failure(f"'{op}' takes real values; take real(), imag(), abs() or phase() first", node)

    def call(self, node):
        fn = node.name
        if fn == "complex":
            if len(node.args) != 2:
                self.error("complex() takes a real and an imaginary part", node)
            return self.lower(_Complex(*node.args))
        if fn == "phase" and len(node.args) == 2:
            (zr, zi), (t, ti) = self.lower(node.args[0]), self.lower(node.args[1])
            if ti is not None:
                self.error("phase() takes a real angle", node)
            if zi is None:
                return O._Binary("*", zr, O._Call("cos", t)), None
            return O._Binary("-", O._Binary("*", zr, O._Call("cos", t)), O._Binary("*", zi, O._Call("sin", t))), None
        parts = [self.lower(a) for a in node.args]
        if fn in ("real", "imag", "conj", "arg") and len(parts) != 1:
            self.error(f"{fn}() takes one value", node)
        if fn == "real":
            return parts[0][0], None
        if fn == "imag":
            r, i = parts[0]
            return (i if i is not None else _zero_like(r)), None
        if fn == "conj":
            r, i = parts[0]
            return r, (None if i is None else O._Unary("-", i))
        if fn == "arg":
            r, i = parts[0]
            return O._Call("atan2", i if i is not None else _zero_like(r), r), None
        if all(i is None for _, i in parts):
            return _like(node, *(r for r, _ in parts)), None
        if fn == "abs" and len(parts) == 1:
            r, i = parts[0]
            return O._Call("sqrt", O._Binary("+", O._Binary("*", r, r), O._Binary("*", i, i))), None
        if fn in ("mag", "mag2") and len(parts) == 1:
            r, i = parts[0]
            m2 = O._Binary("+", O._Call("mag2", r), O._Call("mag2", i))
            return (O._Call("sqrt", m2) if fn == "mag" else m2), None
        if fn == "exp" and len(parts) == 1:
            r, i = parts[0]
            e = O._Call("exp", r)
            return O._Binary("*", e, O._Call("cos", i)), O._Binary("*", e, O._Call("sin", i))
        if fn in _LINEAR:
            r = _like(node, *(r for r, _ in parts))
            i = _like(node, *(i if i is not None else _zero_like(r_) for r_, i in parts))
            return r, i
        if fn in _BILINEAR and len(parts) == 2:
            (ar, ai), (br, bi) = parts

            def f(x, y):
                return O._Call(fn, x, y)

            re = f(ar, br) if ai is None or bi is None else O._Binary("-", f(ar, br), f(ai, bi))
            im = [t for t in ((f(ar, bi) if bi is not None else None), (f(ai, br) if ai is not None else None))
                  if t is not None]
            return re, (im[0] if len(im) == 1 else O._Binary("+", *im))
        if fn == "if" and len(parts) == 3:
            (c, ci), (ar, ai), (br, bi) = parts
            if ci is not None:
                self.error("if() takes a real condition", node)
            return (O._Call("if", c, ar, br),
                    O._Call("if", c, ai if ai is not None else _zero_like(ar), bi if bi is not None else _zero_like(br)))
        raise self.failure(f"{fn}() takes real values; take real(), imag(), abs() or phase(z, angle) first", node)


# --- tensors, component by component ---------------------------------------------

def _sum(terms):
    out = terms[0]
    for t in terms[1:]:
        out = O._Binary("+", out, t)
    return out


class _Tensors(_Pass):
    """Derivatives of tensors written out component by component: div() of a tensor is the
    vector of sums of its entries' gradients, laplacian() of a vector the vector of its
    components' Laplacians, so the derivative compilers see scalars and vectors only."""

    def __init__(self, text, data):
        super().__init__(text)
        self.data = data
        self.bindings = P._BINDINGS.get() or {}
        self.changed = False

    def field_of(self, node):
        if isinstance(node, O._Field):
            return node.field
        name = node.name
        if name in self.bindings:
            return self.bindings[name]
        if self.data is not None and (name in self.data.fields or P._derived(name)):
            return P._named_field(self.data, name)
        return None

    def shape(self, node):
        if isinstance(node, (O._Field, O._Ref)):
            field = self.field_of(node)
            return P._type(field) if field is not None else P.S
        if isinstance(node, O._Name):
            return {"coords": P.V, "iHat": P.V, "jHat": P.V, "kHat": P.V, "identity": P.M}.get(node.name, P.S)
        if isinstance(node, (O._Const, _Quantity)):
            return P.S
        if isinstance(node, (O._Unary, _Declared)):
            return self.shape(node.a)
        if isinstance(node, O._Index):
            return {P.M: P.V, P.V: P.S}.get(self.shape(node.a), P.S)
        if isinstance(node, O._Vector):
            return P.M if all(self.shape(p) == P.V for p in node.parts) else P.V
        if isinstance(node, O._Binary):
            if node.op not in ("+", "-", "*", "/", "%"):
                return P.S
            shapes = {self.shape(node.a), self.shape(node.b)}
            return P.M if P.M in shapes else P.V if P.V in shapes else P.S
        if isinstance(node, O._Call):
            fn, args = node.name, node.args
            first = self.shape(args[0]) if args else P.S
            if fn == "grad":
                return P.V if first == P.S else P.M
            if fn == "div":
                return P.S if first == P.V else P.V
            if fn in ("curl", "norm", "cross", "eigvals"):
                return P.V
            if fn in ("outer", "transpose", "inv", "sym", "skew", "dev"):
                return P.M
            if fn in ("laplacian", "points", "cells", "owner", "neighbor"):
                return first
            if fn == "dot" and len(args) == 2:
                a, b = first, self.shape(args[1])
                return P.M if (a, b) == (P.M, P.M) else P.V if P.M in (a, b) else P.S
            if fn == "if" and len(args) == 3:
                return self.shape(args[1])
            return P.S
        return P.S

    def scaled(self, node, scalar, part, op):
        return O._Binary(op, scalar, part) if op == "*" and scalar is not None else O._Binary(op, part, scalar)

    def comp(self, node, i, j):
        """Entry (i, j) of a tensor-valued node, as a scalar node."""
        if isinstance(node, O._Name) and node.name == "identity":
            return O._Const(1.0 if i == j else 0.0)
        if isinstance(node, O._Vector):
            return self.vcomp(node.parts[i], j)
        if isinstance(node, O._Unary):
            return O._Unary("-", self.comp(node.a, i, j))
        if isinstance(node, _Declared):
            return _Declared(self.comp(node.a, i, j), node.unit)
        if isinstance(node, O._Binary) and node.op in ("+", "-"):
            return O._Binary(node.op, self.comp(node.a, i, j), self.comp(node.b, i, j))
        if isinstance(node, O._Binary) and node.op in ("*", "/"):
            if self.shape(node.a) == P.S and node.op == "*":
                return O._Binary("*", node.a, self.comp(node.b, i, j))
            if self.shape(node.b) == P.S:
                return O._Binary(node.op, self.comp(node.a, i, j), node.b)
        if isinstance(node, O._Call):
            fn, args = node.name, node.args
            if fn == "grad":
                return O._Index(O._Call("grad", self.vcomp(args[0], i)), j)
            if fn == "outer":
                return O._Binary("*", self.vcomp(args[0], i), self.vcomp(args[1], j))
            if fn == "transpose":
                return self.comp(args[0], j, i)
            if fn in ("sym", "skew"):
                return O._Binary("*", O._Const(0.5), O._Binary("+" if fn == "sym" else "-",
                                                                self.comp(args[0], i, j), self.comp(args[0], j, i)))
            if fn == "dev":
                entry = self.comp(args[0], i, j)
                if i != j:
                    return entry
                trace = _sum([self.comp(args[0], k, k) for k in range(3)])
                return O._Binary("-", entry, O._Binary("/", trace, O._Const(3.0)))
            if fn == "dot" and len(args) == 2:
                return _sum([O._Binary("*", self.comp(args[0], i, k), self.comp(args[1], k, j)) for k in range(3)])
            if fn == "if" and len(args) == 3:
                return O._Call("if", args[0], self.comp(args[1], i, j), self.comp(args[2], i, j))
        return O._Index(O._Index(node, i), j)

    def vcomp(self, node, k):
        """Component k of a vector-valued node, as a scalar node."""
        if isinstance(node, O._Name) and node.name in ("iHat", "jHat", "kHat"):
            return O._Const(1.0 if "ijk".index(node.name[0]) == k else 0.0)
        if isinstance(node, O._Vector):
            return node.parts[k]
        if isinstance(node, O._Unary):
            return O._Unary("-", self.vcomp(node.a, k))
        if isinstance(node, _Declared):
            return _Declared(self.vcomp(node.a, k), node.unit)
        if isinstance(node, O._Index) and self.shape(node.a) == P.M:
            return self.comp(node.a, node.k, k)
        if isinstance(node, O._Binary) and node.op in ("+", "-", "*", "/"):
            a = node.a if self.shape(node.a) == P.S else self.vcomp(node.a, k)
            b = node.b if self.shape(node.b) == P.S else self.vcomp(node.b, k)
            return O._Binary(node.op, a, b)
        if isinstance(node, O._Call):
            fn, args = node.name, node.args
            if fn == "dot" and len(args) == 2:
                sa, sb = self.shape(args[0]), self.shape(args[1])
                if sa == P.M and sb == P.V:
                    return _sum([O._Binary("*", self.comp(args[0], k, j), self.vcomp(args[1], j)) for j in range(3)])
                if sa == P.V and sb == P.M:
                    return _sum([O._Binary("*", self.vcomp(args[0], j), self.comp(args[1], j, k)) for j in range(3)])
            if fn == "cross" and len(args) == 2:
                a, b = (k + 1) % 3, (k + 2) % 3
                return O._Binary("-", O._Binary("*", self.vcomp(args[0], a), self.vcomp(args[1], b)),
                                 O._Binary("*", self.vcomp(args[0], b), self.vcomp(args[1], a)))
            if fn == "div" and len(args) == 1 and self.shape(args[0]) == P.M:
                return self.divergence_row(args[0], k)
            if fn == "if" and len(args) == 3:
                return O._Call("if", args[0], self.vcomp(args[1], k), self.vcomp(args[2], k))
        return O._Index(node, k)

    def divergence_row(self, node, i):
        return _sum([O._Index(O._Call("grad", self.comp(node, i, j)), j) for j in range(3)])

    def expand(self, node):
        node = _like(node, *(self.expand(c) for c in _children(node)))
        if isinstance(node, O._Call) and len(node.args) == 1:
            arg = node.args[0]
            if node.name == "div" and self.shape(arg) == P.M:
                self.changed = True
                return _at(O._Vector(*(self.divergence_row(arg, i) for i in range(3))), getattr(node, "pos", 0))
            if node.name == "laplacian" and self.shape(arg) == P.V:
                self.changed = True
                return _at(O._Vector(*(O._Call("laplacian", self.vcomp(arg, i)) for i in range(3))),
                           getattr(node, "pos", 0))
        return node


def expands(source, data):
    """Whether the expression takes derivatives of tensors, written out by component here."""
    try:
        tree = parse(source) if isinstance(source, str) else source
    except P.ExprError:
        return False
    tensors = _Tensors("", data)
    tensors.expand(tree)
    return tensors.changed


# --- units ------------------------------------------------------------------------

_DIMLESS = {"exp", "ln", "log", "log10", "sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh"}
_SAME = {"abs", "ceil", "floor", "transpose", "sym", "skew", "dev", "trace", "eigvals", "vonmises", "mag",
         "points", "cells", "owner", "neighbor", "fsum"}
_NONE = {"sign", "sgn", "not", "norm"}
_PRODUCT = {"dot", "cross", "outer", "ddot"}
_MATCH = {"min", "max", "clamp"}
_DERIVATIVE = {"grad": (1, 1), "div": (1, 1), "curl": (1, 1), "qcrit": (2, 2), "laplacian": (1, 2)}
_NONE_UNIT = "dimensionless"
_BARE = "bare"   # a number in no unit: in a sum or comparison, the unit of what it meets


def _scaled(node, factor, offset=0.0):
    if factor != 1.0:
        node = O._Binary("*", node, O._Const(factor))
    if offset != 0.0:
        node = O._Binary("+", node, O._Const(offset))
    return node


class _Units(_Pass):
    """Each node's unit: a canonical string, None for unknown or _BARE for a plain number."""

    def __init__(self, text, length):
        super().__init__(text)
        self.length = length   # the geometry's unit, None if unknown

    def fail(self, node, error):
        self.error(str(error.args[0]) if error.args else str(error), node)

    def derived(self, node, *terms):
        try:
            return U.derived(*((_NONE_UNIT if u is _BARE else u, k) for u, k in terms))
        except ValueError as error:
            self.fail(node, error)

    def common(self, node, items):
        """Items (node, unit) brought to one unit, the first known one's: (nodes, unit)."""
        target = next((u for _, u in items if u not in (None, _BARE)), None)
        if target is None:
            unit = None if any(u is None for _, u in items) else _BARE
            return [n for n, _ in items], unit
        out = []
        for n, u in items:
            if u in (None, _BARE) or U.same(u, target):
                out.append(n)
                continue
            try:
                factor, offset = U.conversion(u, target)
            except ValueError as error:
                self.error(f"{U.named(u)} and {U.named(target)} do not match: {error.args[0]}", node)
            out.append(_scaled(n, factor, offset))
        unit = None if any(u is None for _, u in items) else target
        return out, unit

    def dimensionless(self, node, what, child, unit):
        """child as a pure number, converted when its unit is one with a scale (mm/m)."""
        if unit in (None, _BARE):
            return child, (None if unit is None else _NONE_UNIT)
        for pure in (_NONE_UNIT, "rad"):   # an angle is a pure number in radians
            try:
                factor, offset = U.conversion(unit, pure)
            except ValueError:
                continue
            return _scaled(child, factor, offset), _NONE_UNIT
        raise self.failure(f"{what} takes a dimensionless value, not {U.named(unit)}", node)

    def check(self, node):
        """(the node with its conversions, its unit)."""
        if isinstance(node, (O._Ref, O._Field)):
            return node, getattr(node, "unit", None)
        if isinstance(node, _Declared):
            return node.a, node.unit
        if isinstance(node, _Quantity):
            return O._Const(node.value), node.unit
        if isinstance(node, O._Const):
            return node, _BARE
        if isinstance(node, O._Name):
            if node.name.startswith("coords"):
                return node, self.length
            return node, _NONE_UNIT
        if isinstance(node, (O._Unary, O._Index)):
            a, u = self.check(node.a)
            return _like(node, a), u
        if isinstance(node, O._Vector):
            parts = [self.check(p) for p in node.parts]
            nodes, unit = self.common(node, parts)
            return _like(node, *nodes), unit
        if isinstance(node, O._Binary):
            return self.binary(node)
        if isinstance(node, O._Call):
            return self.call(node)
        raise self.failure("this expression has no unit rule", node)

    def binary(self, node):
        op = node.op
        a, b = self.check(node.a), self.check(node.b)
        if op in ("+", "-", "%", "<", "<=", ">", ">=", "==", "!="):
            (na, nb), unit = self.common(node, [a, b])
            new = _like(node, na, nb)
            if op in ("+", "-", "%"):
                if op == "-" and unit not in (None, _BARE) and a[1] not in (None, _BARE) and b[1] not in (None, _BARE):
                    unit = U.delta(unit)
                return new, unit
            return new, _NONE_UNIT
        new = _like(node, a[0], b[0])
        if op in ("and", "or"):
            return new, _NONE_UNIT
        if op in ("*", "/"):
            if a[1] is _BARE and b[1] is _BARE:
                return new, _BARE
            return new, self.derived(node, (a[1], 1), (b[1], 1 if op == "*" else -1))
        if op == "^":
            exponent, eu = self.dimensionless(node, "an exponent", b[0], b[1])
            new = _like(node, a[0], exponent)
            if a[1] in (None, _BARE) or a[1] == _NONE_UNIT:
                return new, a[1]
            k = _const(node.b)
            if k is None:
                self.error(f"a power of {U.named(a[1])} needs a constant exponent", node)
            return new, self.derived(node, (a[1], k))
        raise self.failure(f"'{op}' has no unit rule", node)

    def call(self, node):
        fn = node.name
        checked = [self.check(a) for a in node.args]
        nodes = [n for n, _ in checked]
        units = [u for _, u in checked]
        new = _like(node, *nodes)
        if fn in _DERIVATIVE:
            power, length = _DERIVATIVE[fn]
            u = units[0] if units else None
            if u is _BARE:
                u = _NONE_UNIT
            if u is None or self.length is None:
                return new, None
            return new, self.derived(node, (U.delta(u), power), (self.length, -length))
        if fn in _DIMLESS:
            arg, _ = self.dimensionless(node, f"{fn}()", nodes[0], units[0])
            return _like(node, arg), (None if units[0] is None else _NONE_UNIT)
        if fn in _SAME:
            return new, units[0]
        if fn in _NONE:
            return new, _NONE_UNIT
        if fn == "sqrt":
            return new, None if units[0] is None else self.derived(node, (units[0], 0.5))
        if fn == "mag2":
            return new, None if units[0] is None else self.derived(node, (units[0], 2))
        if fn in _PRODUCT:
            return new, self.derived(node, (units[0], 1), (units[1], 1))
        if fn == "det":
            return new, self.derived(node, (units[0], 3))
        if fn == "inv":
            return new, self.derived(node, (units[0], -1))
        if fn in _MATCH:
            parts, unit = self.common(node, checked)
            return _like(node, *parts), unit
        if fn == "atan2":
            parts, _ = self.common(node, checked)
            return _like(node, *parts), _NONE_UNIT
        if fn == "if":
            parts, unit = self.common(node, checked[1:])
            return _like(node, nodes[0], *parts), unit
        if fn == "pow":
            return self.binary(_at(O._Binary("^", *node.args), getattr(node, "pos", 0)))
        raise self.failure(f"units are not known for {fn}()", node)


def needs_front(text, fields, units, data=None):
    """Whether text needs the tree passes: a dictionary, units, a quantity, complex values, or
    a field with a unit."""
    if fields is not None or units is not None or not isinstance(text, str):
        return True
    if _NEEDS.search(text) is not None:
        return True
    if data is None:
        return False
    stored = dict(data.fields.entries())
    for name in set(re.findall(r"[A-Za-z_]\w*", text)):
        field = stored.get(name)
        if field is not None:
            if getattr(field, "unit", None) is not None:
                return True
        elif any(name + r in stored and name + i in stored for r, i in _COMPLEX_SUFFIXES):
            return True
    return False


def needs_tree(tree):
    """An operator expression with complex values, quantities or fields with units in it."""
    return (_has(tree, _Complex) or _has(tree, _Quantity) or _known_units_of_fields(tree)
            or bool(_calls(tree) & {"real", "imag", "conj", "phase", "complex", "arg"}))


def _known_units_of_fields(node):
    if isinstance(node, O._Field):
        return getattr(node.field, "unit", None) is not None
    return any(_known_units_of_fields(c) for c in _children(node))


def prepare(source, data, fields=None, units=None, names=None):
    """(text, {name: td.Field}, unit) for the parser: names bound through fields (a
    FieldDictionary or {name: unit}), complex values split, units checked and converted, and
    with units, the result converted to them. unit is the result's (a canonical string, or
    None for unknown). names: labels already given to fields, shared across expressions."""
    if isinstance(source, str):
        text, tree = source, parse(source)
    else:
        tree = source
        text = type(tree).__name__
        try:
            text = O.Expr.__str__(tree)
        except (AttributeError, TypeError):
            pass
    dictionary = as_dictionary(fields)
    bindings = dictionary.bind(data) if dictionary is not None and data is not None else {}
    tree = _Binder(text, data, bindings).bind(tree)
    tree, imaginary = _Lowering(text).lower(tree)
    if imaginary is not None:
        raise P.ExprError("the result is complex: take real(), imag(), abs(), arg() or phase(z, angle)", 0, text)
    tree = _Tensors(text, data).expand(tree)
    unit = None
    if units is not None or _known_units(tree):
        length = getattr(data, "unit", None)
        if length is None and dictionary is not None and dictionary.length is not None:
            length = U.canonical(dictionary.length)
        tree, unit = _Units(text, length).check(tree)
        if unit is _BARE:
            unit = _NONE_UNIT
        if units is not None:
            try:
                target = U.canonical(units)
            except (ValueError, TypeError) as error:
                raise P.ExprError(str(error), 0, text) from None
            if unit is None:
                raise P.ExprError(f"the result's unit is unknown, so it cannot be given in {target}: give the "
                                  "fields it reads a unit", 0, text)
            try:
                factor, offset = U.conversion(unit, target)
            except ValueError as error:
                raise P.ExprError(f"the result is in {U.named(unit)}, not {target}: {error.args[0]}", 0, text) from None
            tree = _scaled(tree, factor, offset)
            unit = target
    rendered, names = tree.render(data, names)
    return rendered, names, unit
