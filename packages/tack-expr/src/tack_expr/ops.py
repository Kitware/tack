# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Field expressions written with Python operators: ``(U + V / curl(W)).eval(data)``.

Arithmetic, comparisons and NumPy's ufuncs on td.Fields and on the expressions they make
build a tree of operations on the functions the fields represent. ``eval``, ``compile``,
``field``, ``contour`` and ``function`` write it in the calculator syntax over names bound
to its fields for the call, so it compiles through the same parser and code generator as
text: derivatives by the chain rule, conversions, several expressions in one kernel.

A td.Field in an expression is that field, read wherever the expression runs; ``ref(name)``
is whatever ``data.fields[name]`` is when it runs. The dataset passed in gives the topology
and the geometry, and a field must be on its topology. Importing tack_expr gives td.Field
the operators.
"""

import numbers

import numpy as np

import tack.dataset as td

from . import lazy
from . import parser as P


class Expr:
    """An expression of fields: combine with operators, NumPy ufuncs and this module's
    functions; evaluate with ``eval``."""

    __array_priority__ = 1000

    def __add__(self, other):
        return _Binary("+", self, other)

    def __radd__(self, other):
        return _Binary("+", other, self)

    def __sub__(self, other):
        return _Binary("-", self, other)

    def __rsub__(self, other):
        return _Binary("-", other, self)

    def __mul__(self, other):
        return _Binary("*", self, other)

    def __rmul__(self, other):
        return _Binary("*", other, self)

    def __truediv__(self, other):
        return _Binary("/", self, other)

    def __rtruediv__(self, other):
        return _Binary("/", other, self)

    def __pow__(self, other):
        return _Binary("^", self, other)

    def __rpow__(self, other):
        return _Binary("^", other, self)

    def __mod__(self, other):
        return _Binary("%", self, other)

    def __neg__(self):
        return _Unary("-", self)

    def __pos__(self):
        return self

    def __abs__(self):
        return _Call("abs", self)

    def __lt__(self, other):
        return _Binary("<", self, other)

    def __le__(self, other):
        return _Binary("<=", self, other)

    def __gt__(self, other):
        return _Binary(">", self, other)

    def __ge__(self, other):
        return _Binary(">=", self, other)

    def __and__(self, other):
        return _Binary("and", self, other)

    def __rand__(self, other):
        return _Binary("and", other, self)

    def __or__(self, other):
        return _Binary("or", self, other)

    def __ror__(self, other):
        return _Binary("or", other, self)

    def __invert__(self):
        return _Call("not", self)

    def __getitem__(self, k):
        if isinstance(k, tuple):
            out = self
            for i in k:
                out = out[i]
            return out
        if k not in (0, 1, 2):
            raise IndexError("a vector has components 0, 1 and 2, and a tensor rows 0, 1 and 2")
        return _Index(self, k)

    @property
    def x(self):
        return _Index(self, 0)

    @property
    def y(self):
        return _Index(self, 1)

    @property
    def z(self):
        return _Index(self, 2)

    def __bool__(self):
        raise TypeError("an expression has no truth value until it is evaluated: combine conditions "
                        "with &, | and ~, and choose with where(condition, a, b)")

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        name = _UFUNCS.get(ufunc.__name__)
        if method != "__call__" or kwargs or name is None:
            return NotImplemented
        if name in _BINARY_UFUNCS:
            return _Binary(name, *inputs)
        if name == "neg":
            return _Unary("-", inputs[0])
        return _Call(name, *inputs)

    # --- evaluation -------------------------------------------------------

    def _prepare(self, data, fields=None, units=None):
        """(text, names, unit): through the tree passes when fields, units or complex values
        call for them."""
        from . import _front
        if fields is not None or units is not None or _front.needs_tree(self):
            return _front.prepare(self, data, fields, units)
        return (*self.render(data), None)

    def eval(self, data, at="cells", replace_invalid=None, name=None, project=False, fields=None, units=None):
        """The expression at ``at`` ("cells", "points", "faces" or a space of ``data``'s), a
        td.Field; with ``name``, also put in ``data.fields``. fields and units: as
        tack_expr.evaluate's."""
        from . import _front
        if fields is not None or units is not None or _front.needs_tree(self):
            return P.evaluate(self, data, at=at, replace_invalid=replace_invalid, name=name, project=project,
                              fields=fields, units=units)
        text, names = self.render(data)
        with P.bound(names):
            return P.evaluate(text, data, at=at, replace_invalid=replace_invalid, name=name, project=project)

    def eval_complex(self, data, at="cells", replace_invalid=None, fields=None, units=None, project=False):
        """The real and imaginary parts of a complex expression, (real, imag) td.Fields, from one kernel."""
        return P.evaluate_complex(self, data, at, replace_invalid, fields, units, project)

    def compile(self, data, at="cells", replace_invalid=None, fields=None, units=None):
        """A Compiled kernel, called as ``compiled(data)`` as often as the values change."""
        text, names, unit = self._prepare(data, fields, units)
        with P.bound(names):
            compiled = P.compile_expression(text, data, at, replace_invalid)
        compiled.unit = unit
        return compiled

    def field(self, data, fields=None, units=None):
        """A td.Field computed wherever an algorithm reads it, stored nowhere; its unit the
        expression's."""
        text, names, unit = self._prepare(data, fields, units)
        with P.bound(names):
            f = lazy.field(text, data)
        return td.Field(f.space, f.values, unit) if unit is not None else f

    def function(self, data, at="cells", fields=None, units=None):
        """The expression as a device class for kernels of your own (tack_expr.for_each)."""
        text, names, _ = self._prepare(data, fields, units)
        with P.bound(names):
            return P.function(text, data, at)

    def render(self, data=None, names=None):
        """The calculator text and the fields its names stand for. A field is named as
        ``data.fields`` names it where that name is its alone, otherwise ``_f0``, ``_f1``..."""
        leaves = []
        self._leaves(leaves)
        refs = {leaf.name for leaf in leaves if isinstance(leaf, _Ref)}
        names = {} if names is None else names
        by_field = {id(f): n for n, f in names.items()}
        for leaf in leaves:
            if not isinstance(leaf, _Field) or id(leaf.field) in by_field:
                continue
            if isinstance(data, td.PartitionedDataSet):
                raise ValueError("a td.Field is one dataset's; over pieces, name the fields with "
                                 "tack_expr.ref(name), read from each piece")
            if data is not None and leaf.field.space.topology is not data.topology:
                raise ValueError(f"{leaf.field!r} is on another topology than the dataset's: an "
                                 "expression is evaluated on the topology and geometry of the dataset "
                                 "passed in")
            label = _stored_name(data, leaf.field)
            if label is None or label in refs or label in names:
                label = f"_f{len(names)}"
            names[label] = leaf.field
            by_field[id(leaf.field)] = label
        return self._text(by_field), names

    def __str__(self):
        return self.render()[0]

    def __repr__(self):
        return f"tack_expr expression {self}"


def _stored_name(data, field):
    if data is None or isinstance(data, td.PartitionedDataSet):
        return None
    found = [name for name, f in data.fields.entries() if f is field]
    if len(found) != 1:
        return None
    name = found[0]
    if sum(1 for n, _ in data.fields.entries() if n == name) != 1:
        return None
    return name


def _wrap(value):
    if isinstance(value, Expr):
        return value
    if isinstance(value, td.Field):
        return _Field(value)
    if isinstance(value, bool):
        return _Const(1.0 if value else 0.0)
    if isinstance(value, numbers.Real):
        return _Const(float(value))
    if isinstance(value, numbers.Complex):
        from ._front import _Complex
        return _Complex(float(value.real), float(value.imag))
    raise TypeError(f"an expression combines fields, expressions and numbers, not {type(value).__name__}")


class _Field(Expr):
    def __init__(self, field):
        self.field = field

    def _leaves(self, out):
        out.append(self)

    def _text(self, names):
        return names.get(id(self.field), "_f")


class _Ref(Expr):
    def __init__(self, name):
        self.name = name

    def _leaves(self, out):
        out.append(self)

    def _text(self, names):
        return self.name if self.name.isidentifier() else f'"{self.name}"'


class _Const(Expr):
    def __init__(self, value):
        if not np.isfinite(value):
            raise ValueError(f"a constant is finite, not {value}")
        self.value = value

    def _leaves(self, out):
        pass

    def _text(self, names):
        text = repr(self.value)
        return f"({text})" if self.value < 0 else text


class _Name(Expr):
    """A name of the syntax itself: coords, pi."""

    def __init__(self, name):
        self.name = name

    def _leaves(self, out):
        pass

    def _text(self, names):
        return self.name


class _Binary(Expr):
    def __init__(self, op, a, b):
        self.op, self.a, self.b = op, _wrap(a), _wrap(b)

    def _leaves(self, out):
        self.a._leaves(out)
        self.b._leaves(out)

    def _text(self, names):
        return f"({self.a._text(names)} {self.op} {self.b._text(names)})"


class _Unary(Expr):
    def __init__(self, op, a):
        self.op, self.a = op, _wrap(a)

    def _leaves(self, out):
        self.a._leaves(out)

    def _text(self, names):
        return f"({self.op}{self.a._text(names)})"


class _Call(Expr):
    def __init__(self, name, *args):
        self.name, self.args = name, [_wrap(a) for a in args]

    def _leaves(self, out):
        for a in self.args:
            a._leaves(out)

    def _text(self, names):
        return f"{self.name}({', '.join(a._text(names) for a in self.args)})"


class _Index(Expr):
    def __init__(self, a, k):
        self.a, self.k = _wrap(a), k

    def _leaves(self, out):
        self.a._leaves(out)

    def _text(self, names):
        return f"{self.a._text(names)}[{self.k}]"


class _Vector(Expr):
    def __init__(self, *parts):
        self.parts = [_wrap(p) for p in parts]

    def _leaves(self, out):
        for p in self.parts:
            p._leaves(out)

    def _text(self, names):
        return "{" + ", ".join(p._text(names) for p in self.parts) + "}"


def _field_only(fn):
    """An operator whose argument is a field itself, read on other entities."""
    def operator(f):
        leaf = _wrap(f)
        if not isinstance(leaf, (_Field, _Ref)):
            raise TypeError(f"{fn}() takes a field, not an expression: evaluate the expression into a "
                            "field first")
        return _Call(fn, leaf)
    operator.__name__ = fn
    operator.__doc__ = f"The calculator's {fn}() of a field."
    return operator


def _function(fn, doc):
    def call(*args):
        return _Call(fn, *args)
    call.__name__ = fn
    call.__doc__ = doc
    return call


def ref(name):
    """The field ``data.fields[name]`` of whatever dataset the expression runs on."""
    return _Ref(name)


def vec(a, b, c):
    """A vector of three scalar expressions."""
    return _Vector(a, b, c)


def where(condition, a, b):
    """``a`` where ``condition`` is nonzero, else ``b``."""
    return _Call("if", condition, a, b)


coords = _Name("coords")
pi = _Name("pi")

grad = _function("grad", "The world gradient of a scalar, or the Jacobian rows of a vector.")
div = _function("div", "The divergence of a vector.")
curl = _function("curl", "The curl of a vector.")
qcrit = _function("qcrit", "The Q-criterion of a velocity.")
laplacian = _function("laplacian", "The Laplacian of a scalar.")
mag = _function("mag", "A vector's magnitude.")
mag2 = _function("mag2", "A vector's squared magnitude.")
norm = _function("norm", "A vector divided by its magnitude.")
dot = _function("dot", "The dot product of two vectors.")
cross = _function("cross", "The cross product of two vectors.")
clamp = _function("clamp", "clamp(lo, x, hi).")
minimum = _function("min", "The smaller of two scalars.")
trace = _function("trace", "A tensor's trace.")
det = _function("det", "A tensor's determinant.")
transpose = _function("transpose", "A tensor's transpose.")
inv = _function("inv", "A tensor's inverse.")
sym = _function("sym", "A tensor's symmetric part.")
skew = _function("skew", "A tensor's antisymmetric part.")
dev = _function("dev", "A tensor's deviatoric part, less a third of its trace on the diagonal.")
outer = _function("outer", "The tensor product of two vectors.")
ddot = _function("ddot", "The double contraction of two tensors, A:B.")
vonmises = _function("vonmises", "The von Mises equivalent of a stress tensor.")
eigvals = _function("eigvals", "The eigenvalues of a tensor's symmetric part, largest first.")
identity = _Name("identity")
real = _function("real", "A complex value's real part.")
imag = _function("imag", "A complex value's imaginary part.")
conj = _function("conj", "A complex value's conjugate.")
arg = _function("arg", "A complex value's angle, atan2(imag, real).")
phase = _function("phase", "phase(z, angle): the real part of z exp(i angle), a harmonic response at that phase.")
maximum = _function("max", "The larger of two scalars.")
cells = _field_only("cells")
points = _field_only("points")
owner = _field_only("owner")
neighbor = _field_only("neighbor")
fsum = _field_only("fsum")

_UFUNCS = {"add": "+", "subtract": "-", "multiply": "*", "divide": "/", "true_divide": "/", "power": "^",
           "remainder": "%", "less": "<", "less_equal": "<=", "greater": ">", "greater_equal": ">=",
           "negative": "neg", "absolute": "abs", "sqrt": "sqrt", "exp": "exp", "log": "ln", "log10": "log10",
           "sin": "sin", "cos": "cos", "tan": "tan", "arcsin": "asin", "arccos": "acos", "arctan": "atan",
           "arctan2": "atan2", "sinh": "sinh", "cosh": "cosh", "tanh": "tanh", "floor": "floor", "ceil": "ceil",
           "sign": "sign", "minimum": "min", "maximum": "max"}
_BINARY_UFUNCS = {"+", "-", "*", "/", "^", "%", "<", "<=", ">", ">="}


def compile_many(expressions, data, at="cells", replace_invalid=None, fields=None, units=None):
    """tack_expr.compile_many of a mapping of names to expressions or calculator text, sharing
    the bound fields' names. fields and units: as tack_expr.compile_many's."""
    from . import _front
    trees = {key: _wrap(e) if not isinstance(e, str) else e for key, e in expressions.items()}
    if fields is not None or units is not None or any(
            not isinstance(e, str) and _front.needs_tree(e) for e in trees.values()):
        return P.compile_many(trees, data, at, replace_invalid, fields, units)
    texts, names = {}, {}
    for key, e in expressions.items():
        if isinstance(e, str):
            texts[key] = e
        else:
            texts[key], names = _wrap(e).render(data, names)
    with P.bound(names):
        return P.compile_many(texts, data, at, replace_invalid)


def _install():
    """td.Field's operators: each makes the expression of the field it is applied to."""
    def lift(method):
        def lifted(self, *args, **kwargs):
            return getattr(_Field(self), method)(*args, **kwargs)
        lifted.__name__ = method
        return lifted

    for method in ("__add__", "__radd__", "__sub__", "__rsub__", "__mul__", "__rmul__", "__truediv__",
                   "__rtruediv__", "__pow__", "__rpow__", "__mod__", "__neg__", "__pos__", "__abs__",
                   "__lt__", "__le__", "__gt__", "__ge__", "__and__", "__rand__", "__or__", "__ror__",
                   "__invert__", "__getitem__", "__array_ufunc__"):
        if method not in td.Field.__dict__:
            setattr(td.Field, method, lift(method))
    td.Field.__array_priority__ = 1000


_install()
