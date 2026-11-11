# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Compile VTK array calculator expressions over tack.dataset fields to Tack kernels.

An expression is evaluated at an output location:

  "cells"   one value per cell, at the cell's parametric center
  "points"  one value per point
  "faces"   one value per face
  a space   an L2 space (any shape with a reference element), or an H1 space
            for an expression of continuous fields: one value per node

Fields are tack.dataset fields named in data.fields, or bound to names for a
compilation by bound(), which the operator expressions of ops use. A field with a basis is
evaluated through its basis at the output's parametric coordinate; values on
cells, points or faces are read by entity. Data on other entities needs an
explicit operator: points(), cells(), owner(), neighbor() or fsum().

The parser checks scalar/vector types and where each field can be read, and
emits a Tack kernel, one assignment per operation. The source depends only on
the expression's structure and how its fields are read: constants are kernel
arguments, and no user text reaches it. It is written to a module in a cache
directory and imported, so @tack.kernel compiles it like any other kernel.
"""

import contextlib
import contextvars
import hashlib
import importlib.util
import os
import re
import time

import numpy as np

import tack
import tack.dataset as td
from tack.arrays import materialize

from ._helpers import component, entry_cells, gather, position_cells

try:  # public in a Tack with the world_gradient export
    from tack.dataset import world_gradient as _world_gradient_fn  # noqa: F401
    _GRADIENT_IMPORT = "from tack.dataset import world_gradient as _world_gradient\n"
except ImportError:
    _GRADIENT_IMPORT = "from tack.dataset.algorithms.gradients import _world_gradient\n"

S, V, M = "scalar", "vector", "tensor"

CACHE_DIR = os.environ.get("TACK_EXPR_CACHE",
                           os.path.join(os.path.expanduser("~"), ".cache", "tack_expr"))
# Generated modules unused for this many days are removed when tack_expr is imported (0: never).
CACHE_DAYS = float(os.environ.get("TACK_EXPR_CACHE_DAYS", "30"))


_BINDINGS = contextvars.ContextVar("tack_expr_bindings", default=None)


@contextlib.contextmanager
def bound(fields):
    """Within it, names in the expressions compiled resolve to these td.Fields before
    data.fields', and what is compiled keeps reading them."""
    token = _BINDINGS.set(dict(fields))
    try:
        yield
    finally:
        _BINDINGS.reset(token)


class ExprError(ValueError):
    def __init__(self, message, pos, text):
        super().__init__(f"{message} at column {pos + 1}\n  {text}\n  {' ' * pos}^")
        self.pos = pos


class _NeedCorners(Exception):
    """A point expression reads a field that needs the cell-corner loop."""


_TOKEN = re.compile(r"""\s*(?:
    (?P<num>(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)
  | (?P<name>[A-Za-z_][A-Za-z0-9_]*)
  | (?P<str>"[^"]*")
  | (?P<op>&&|\|\||==|!=|<>|<=|>=|[-+*/%^()\[\]{},?:<>=!&|])
)""", re.VERBOSE)

_UNARY_S = {
    "abs": "abs", "acos": "acos", "asin": "asin", "atan": "atan", "ceil": "ceil",
    "cos": "cos", "cosh": "cosh", "exp": "exp", "floor": "floor", "ln": "log",
    "log": "log", "log10": "log10", "sin": "sin", "sinh": "sinh", "sqrt": "sqrt",
    "tan": "tan", "tanh": "tanh", "sign": "tm.sign", "sgn": "tm.sign",
}
_COMPARE = {"<": "<", "<=": "<=", ">": ">", ">=": ">=",
            "==": "==", "=": "==", "!=": "!=", "<>": "!="}
_INFIX = {  # token: (left binding power, right-associative)
    "?": (10, True),
    "or": (20, False), "||": (20, False), "|": (20, False),
    "and": (30, False), "&&": (30, False), "&": (30, False),
    "==": (40, False), "=": (40, False), "!=": (40, False), "<>": (40, False),
    "<": (50, False), "<=": (50, False), ">": (50, False), ">=": (50, False),
    "+": (60, False), "-": (60, False),
    "*": (70, False), "/": (70, False), "%": (70, False),
    "^": (90, True),
    "[": (100, False),
}
_PREFIX_BP = 80
_FIELD_FUNCTIONS = ("grad", "div", "curl", "qcrit", "points", "cells", "owner", "neighbor", "fsum")
# modes: cells, nodes, corners (points through cell corners), pointloop (points), faces
_CELL_MODES = ("cells", "nodes", "corners")


def _kind(field):
    """Where a field's values are: 'basis' (H1, L2, Constant), 'values-<entity>', or
    'hdiv'/'hcurl' (vector spaces read through a Piola map)."""
    space = field.space
    if isinstance(space, td.Values):
        return f"values-{space.on}"
    if isinstance(space, (td.H1, td.L2, td.Constant)):
        return "basis"
    if type(space).__name__ in ("HDiv", "HCurl"):
        return type(space).__name__.lower()
    return type(space).__name__


def _is_vector(field):
    if _kind(field) in ("hdiv", "hcurl"):
        return True
    values = field.values
    return getattr(values, "scalar_size", 0) == 3 * getattr(values, "size", -1)


def _type(field):
    """S, V or M: M a 3x3 tensor, stored as mat3 or as 9 or 6 (symmetric: XX YY ZZ XY YZ XZ)
    components."""
    if _kind(field) in ("hdiv", "hcurl"):
        return V
    values = field.values
    size = getattr(values, "size", 0) or 1
    return {3: V, 6: M, 9: M}.get(getattr(values, "scalar_size", size) // size, S)


def _as_matrix(field, code):
    """A tensor field's value as a mat3, from the components it is stored as."""
    if getattr(getattr(field.values, "dtype", None), "shape", None) == (3, 3):
        return None
    n = field.values.scalar_size // field.values.size
    c = [f"{code}[{k}]" for k in range(n)]
    if n == 9:
        return f"tack.mat3({', '.join(c)})"
    xx, yy, zz, xy, yz, xz = c
    return f"tack.mat3({xx}, {xy}, {xz}, {xy}, {yy}, {yz}, {xz}, {yz}, {zz})"


def _entity_numbered_h1(field):
    """An H1 field whose first num_points values are its point values."""
    return isinstance(field.space, td.H1) and getattr(field.space, "dofs", None) is None


# Where each mode reads a field: a mesh quantity (_MESH) is the one there.
_AT = {"cells": "cells", "corners": "cells", "faces": "faces", "nodes": "points",
       "pointloop": "points"}


def _surface_normals(name, on):
    def make(data):
        made = td.surface_normals(data).fields[name]
        return td.Field(td.Values(data, on), made.values, made.unit)
    return make


#: The mesh quantities an expression names where the dataset stores no field of the
#: name, by where they are read: Tack's geometry functions, which keep them with the
#: mesh, so they are computed once a mesh.
_MESH = {
    "normals": {"points": _surface_normals("normals", "points"),
                "cells": _surface_normals("cell_normals", "cells"),
                "faces": lambda data: td.face_geometry(data).normal},
    "measures": {"cells": lambda data: td.cell_geometry(data).measure,
                 "faces": lambda data: td.face_geometry(data).area},
    "centroids": {"cells": lambda data: td.cell_geometry(data).centroid},
}


def _location(field):
    space = field.space
    if isinstance(space, td.H1) or (isinstance(space, td.Values) and space.on == "points"):
        return "points"
    if isinstance(space, (td.Constant, td.L2)) or (isinstance(space, td.Values) and space.on == "cells"):
        return "cells"
    if isinstance(space, td.Values) and space.on == "faces":
        return "faces"
    return None


def _named_field(data, name, at=None):
    """The field a name stands for on data, read at ``at`` (points, cells, faces, or None
    for wherever it is): a field stored under the name there comes first, then the mesh
    quantity of the name there (_MESH: normals, measures, centroids), then a field stored
    under the name elsewhere. With no ``at``, the stored field, else the quantity at the
    first location the dataset has it."""
    if name not in _MESH:
        return data.fields[name]
    stored = data.fields.named(name)
    places = _MESH[name]
    if at is not None:
        for field in stored:
            if _location(field) == at:
                return field
        return places[at](data) if at in places else data.fields[name]
    if stored:
        return data.fields[name]
    reasons = []
    for where, make in places.items():
        try:
            return make(data)
        except (ValueError, TypeError, NotImplementedError) as e:
            reasons.append(f"at {where}: {e}")
    raise KeyError(f"{name!r} is no part of this dataset ({'; '.join(reasons)})")


def _derived(name):
    return name in _MESH


class _Compiler:
    def __init__(self, text, data, mode, continuous_only=False):
        self.text, self.data, self.mode = text, data, mode
        self.continuous_only = continuous_only
        self.reference = getattr(data.topology, "reference_cells", True)
        self.toks = self._tokenize(text)
        self.k = 0
        self.lines = []
        self.args = []           # (kind, payload): 'field' -> a view in the kernel, 'array' raw
        self.arg_index = {}
        self.loaded = {}
        self.consts = []
        self.uses_world_gradient = False
        self.uses_vector_spaces = False
        self.uses_derivatives = False
        self.uses_hcurl_curl = False
        self.uses_hdiv_div = False
        self.fields_read = set()
        self.uses_pc = False
        self.bindings = _BINDINGS.get() or {}
        self.matrices = set()
        self.toks = self._expand(self.toks, ())

    def _expand(self, toks, within):
        """The tokens with each name of a stored expression (``fx.field``) replaced by its
        expression in parentheses, its own names included, so it compiles into the same
        kernel with its derivatives by the chain rule and binds when evaluated. A name a
        conversion takes (``points(T2)``) stays a name."""
        out = []
        for i, (kind, val, pos) in enumerate(toks):
            name = val[1:-1] if kind == "str" else val if kind == "name" else None
            before = [t[1] for t in out[-2:]]
            takes_name = len(before) == 2 and before[1] == "(" and before[0] in _TAKES_A_NAME
            stored = None
            if name and not takes_name and (i + 1 == len(toks) or toks[i + 1][1] != "("):
                stored = _stored_expression(self._named(name))
            if stored is None:
                out.append((kind, val, pos))
                continue
            if name in within:
                raise ExprError(f"'{name}' is defined in terms of itself", pos, self.text)
            self.bindings = {**self.bindings, **getattr(stored.space, "names", {})}
            inner = self._tokenize(stored.space.text)[:-1]
            out += [("op", "(", pos), *[(k, v, pos) for k, v, _ in
                                        self._expand(inner, (*within, name))], ("op", ")", pos)]
        return out

    def _named(self, name):
        if name in self.bindings:
            return self.bindings[name]
        if _derived(name) and name not in self.data.fields:
            return None
        try:
            return self.data.fields[name]
        except (KeyError, ValueError, TypeError):
            return None

    # --- emission -------------------------------------------------------

    def emit(self, code, t):
        name = f"t{len(self.lines)}"
        self.lines.append(f"{name} = {code}")
        return name, t

    def arg(self, kind, key, payload):
        if (kind, key) not in self.arg_index:
            self.arg_index[(kind, key)] = len(self.args)
            self.args.append((kind, payload))
        return f"A{self.arg_index[(kind, key)]}"

    def raw(self, key, getter):
        """A raw array argument: getter(data, memo) gives a field (or array) at each call."""
        def values(d, m, get=getter):
            got = get(d, m)
            return materialize(getattr(got, "values", got))
        return self.arg("array", key, values)

    def by_name(self, name):
        if name in self.bindings:
            field = self.bindings[name]
            return lambda d, m: field
        at = _AT.get(self.mode)
        return lambda d, m: _named_field(d, name, at)

    def has_field(self, name):
        return name in self.bindings or name in self.data.fields or _derived(name)

    @staticmethod
    def once(key, make):
        """A pre-pass run once per call, shared by every argument that needs it."""
        def get(d, m):
            if key not in m:
                m[key] = make(d, m)
            return m[key]
        return get

    def const(self, value):
        self.consts.append(float(value))
        return f"c{len(self.consts) - 1}", S

    def const_value(self, code):
        if code.startswith("c") and code[1:].isdigit():
            return self.consts[int(code[1:])]
        return None

    def world_gradient(self, u, pc):
        """Code for the world gradient of field view u, which its view gives."""
        return f"_world_gradient(dom, {u}, c, {pc})"

    def load(self, key, code, t):
        if key not in self.loaded:
            local = f"a{len(self.loaded)}"
            self.lines.append(f"{local} = {code}")
            self.loaded[key] = local
        return self.loaded[key], t

    def pc(self, pos):
        if not self.reference:
            self.error("polyhedral cells have no parametric coordinates; use cells() for "
                       "point data", pos)
        self.uses_pc = True
        return "pc"

    def num_points(self):
        return "dom.NUM_POINTS" if self.reference else "dom.num_points(c)"

    # --- tokens ---------------------------------------------------------

    def _tokenize(self, text):
        toks, pos = [], 0
        while text[pos:].strip():
            m = _TOKEN.match(text, pos)
            if not m:
                raise ExprError("unexpected character", len(text) - len(text[pos:].lstrip()), text)
            kind = m.lastgroup
            val = m.group(kind)
            if kind == "name" and val in ("and", "or", "not"):
                kind = "op"
            toks.append((kind, val, m.start(kind)))
            pos = m.end()
        toks.append(("end", "", len(text)))
        return toks

    def peek(self):
        return self.toks[self.k]

    def take(self):
        tok = self.toks[self.k]
        self.k += 1
        return tok

    def expect(self, val):
        _, v, pos = self.take()
        if v != val:
            self.error(f"expected '{val}'", pos)

    def error(self, message, pos):
        raise ExprError(message, pos, self.text)

    def failure(self, message, pos):
        """The error for message at pos, for a caller to raise."""
        return ExprError(message, pos, self.text)

    def need(self, got, want, pos):
        if got != want:
            self.error(f"expected a {want}, got a {got}", pos)

    # --- fields ---------------------------------------------------------

    def lookup(self, name, pos):
        if not self.has_field(name):
            self.error(f"unknown field '{name}'", pos)
        self.fields_read.add(name)
        return (self.bindings[name] if name in self.bindings
                else _named_field(self.data, name, _AT.get(self.mode)))

    def field(self, name, pos, field=None, key=None, getter=None):
        """A field read at the output location, or an error saying what to use."""
        field = field if field is not None else self.lookup(name, pos)
        return self.matrix(field, self._read(name, pos, field, key, getter), key or name)

    def matrix(self, field, read, key):
        """read (code, type) of field, a tensor stored as components given as a mat3."""
        code, t = read
        if t != M or code in self.matrices:
            return read
        made = _as_matrix(field, code)
        if made is None:
            return read
        code, _ = self.load(f"mat:{key}", made, M)
        self.matrices.add(code)
        return code, M

    def _read(self, name, pos, field, key, getter):
        key = key or name
        getter = getter or self.by_name(name)
        t = _type(field)
        kind = _kind(field)
        mode = self.mode
        constant = kind == "values-cells" or isinstance(field.space, td.Constant)
        if self.continuous_only and kind == "basis" and not isinstance(field.space, td.H1):
            self.error(f"'{name}' is not continuous, and the output space is H1; evaluate into "
                       "an L2 space and td.project the result into H1, or pass project=True", pos)
        if mode in ("cells", "nodes") and constant:
            return self.load(key, f"{self.raw(key, getter)}[dom.entity_id(c)]", t)
        if kind in ("hdiv", "hcurl"):
            if mode not in ("cells", "nodes"):
                self.error(f"'{name}' is an {type(field.space).__name__} field, read inside cells; "
                           "evaluate it at cells or at an L2 space's nodes", pos)
            self.uses_vector_spaces = True
            u = self.arg("field", key, getter)
            return self.load(key, f"{kind}_value(dom, {u}, c, {self.pc(pos)})", V)
        if mode in _CELL_MODES and kind == "basis":
            if mode == "corners" and not isinstance(field.space, td.H1):
                self.error(f"'{name}' is not continuous; use points({name}) for point values", pos)
            if not self.reference:
                if mode == "corners" and _entity_numbered_h1(field):
                    return self.load(key, f"{self.raw(key, getter)}[dom.point_id(c, j)]", t)
                self.error(f"'{name}' has a basis, and polyhedral cells have no parametric "
                           f"coordinates; use cells({name})", pos)
            return self.load(key, f"{self.arg('field', key, getter)}.value(c, {self.pc(pos)})", t)
        if mode == "pointloop":
            if kind == "values-points" or (kind == "basis" and isinstance(field.space, td.H1)):
                if kind == "basis" and not _entity_numbered_h1(field):
                    raise _NeedCorners()
                return self.load(key, f"{self.raw(key, getter)}[p]", t)
        if mode == "corners" and kind == "values-points":
            return self.load(key, f"{self.raw(key, getter)}[dom.point_id(c, j)]", t)
        if mode == "faces" and kind == "values-faces":
            return self.load(key, f"{self.raw(key, getter)}[e]", t)
        where = {"basis": "cells", "values-cells": "cells", "values-points": "points",
                 "values-faces": "faces", "values-edges": "edges"}.get(kind, kind)
        here = {"pointloop": "points", "corners": "points"}.get(mode, mode)
        hint = {("faces", "cells"): f"owner({name}) or neighbor({name})",
                ("cells", "faces"): f"fsum({name})",
                ("nodes", "faces"): f"fsum({name})",
                ("cells", "points"): f"cells({name})",
                ("nodes", "points"): f"cells({name})",
                ("points", "cells"): f"points({name})"}.get((here, where), "a conversion")
        if here == "points" and kind == "basis":
            self.error(f"'{name}' is not continuous; use points({name}) for point values", pos)
        raise self.failure(f"'{name}' is on {where}, the expression is on {here}; use {hint}", pos)

    def cells_of(self, name, pos):
        """cells(T): each cell's mean of its corner values, inside the kernel."""
        if self.mode not in ("cells", "nodes"):
            self.error("cells() is for cell and node expressions", pos)
        field = self.lookup(name, pos)
        kind = _kind(field)
        t = _type(field)
        if kind == "values-cells" or isinstance(field.space, td.Constant):
            return self.field(name, pos)
        if kind == "values-points" or _entity_numbered_h1(field):
            read = f"{self.raw(name, self.by_name(name))}[dom.point_id(c, j{len(self.lines)})]"
        elif kind == "basis" and self.reference:
            read = f"{self.arg('field', name, self.by_name(name))}.corner_value(c, j{len(self.lines)})"
        else:
            self.error("cells() takes point data or a field with a basis", pos)
        k = len(self.lines)
        n = self.num_points()
        self.lines += [f"m{k} = {read.replace(f'j{k}', '0')}",
                       f"for j{k} in range(1, {n}):",
                       f"    m{k} += {read}"]
        return self.load(f"cells:{name}", f"m{k} / float({n})", t)

    def points_of(self, name, pos):
        """points(T): each point's mean over the cells around it."""
        if self.mode not in ("pointloop", "corners"):
            self.error("points() is for point expressions", pos)
        field = self.lookup(name, pos)
        kind = _kind(field)
        t = _type(field)
        if kind == "values-points" or isinstance(field.space, td.H1):
            return self.field(name, pos)
        if kind not in ("basis", "values-cells"):
            self.error("points() takes cell data or a field with a basis", pos)
        constant = kind == "values-cells" or isinstance(field.space, td.Constant)
        pid = "p" if self.mode == "pointloop" else "dom.point_id(c, j)"
        if constant and _grid_dims(self.data) is not None:
            return self.points_on_grid(name, pid, t)
        links = _point_links(self.data) if constant else None
        if links is None:
            # A pass through td.project, which averages any source.
            projected = td.project(self.data, field, td.H1(self.data))
            get = self.once(f"points:{name}",
                            lambda d, m: td.project(d, d.fields[name], td.H1(d)))
            return self.field(name, pos, projected, f"points:{name}", get)
        offsets, cells_of_entry = links
        lo = self.raw("links:offsets", lambda d, m: offsets)
        ec = self.raw("links:cells", lambda d, m: cells_of_entry)
        a = self.raw(name, self.by_name(name))
        k = len(self.lines)
        self.lines += [f"q{k} = {pid}",
                       f"s{k} = 0.0 * {a}[0]",
                       f"for e{k} in range({lo}[q{k}], {lo}[q{k} + 1]):",
                       f"    s{k} += {a}[{ec}[e{k}]]"]
        return self.load(f"points:{name}", f"s{k} / float(max({lo}[q{k} + 1] - {lo}[q{k}], 1))", t)

    def points_on_grid(self, name, pid, t):
        """points(T) of cell data on a structured grid: the mean over the up to 8 cells around
        each point, found by index arithmetic."""
        D = self.raw("grid:dims", lambda d, m: _grid_dims(d))
        a = self.raw(name, self.by_name(name))
        k = len(self.lines)
        ok = " * ".join(f"tack.select({v}{k} {op} {bound}, 1.0, 0.0)"
                        for v, i in (("gx", 3), ("gy", 4), ("gz", 5))
                        for op, bound in ((">=", "0"), ("<", f"{D}[{i}]")))
        self.lines += [
            f"q{k} = {pid}",
            f"ix{k} = q{k} % {D}[0]",
            f"iy{k} = (q{k} // {D}[0]) % {D}[1]",
            f"iz{k} = q{k} // ({D}[0] * {D}[1])",
            f"s{k} = 0.0 * {a}[0]",
            f"w{k} = 0.0",
            f"for n{k} in range(8):",
            f"    gx{k} = ix{k} - 1 + n{k} % 2",
            f"    gy{k} = iy{k} - 1 + (n{k} // 2) % 2",
            f"    gz{k} = iz{k} - 1 + n{k} // 4",
            f"    ok{k} = {ok}",
            (f"    cell{k} = min(max(gx{k}, 0), {D}[3] - 1) + {D}[3] * (min(max(gy{k}, 0), {D}[4] - 1)"
             f" + {D}[4] * min(max(gz{k}, 0), {D}[5] - 1))"),
            f"    s{k} += ok{k} * {a}[cell{k}]",
            f"    w{k} += ok{k}",
        ]
        return self.load(f"points:{name}", f"s{k} / w{k}", t)

    def side(self, fn, name, pos):
        """owner(T), neighbor(T): a cell value on side 0 or 1 of the face."""
        if self.mode != "faces":
            self.error(f"{fn}() is for face expressions", pos)
        field = self.lookup(name, pos)
        get = self.by_name(name)
        if _kind(field) == "basis" and not isinstance(field.space, td.Constant):
            get = self.once(f"cells:{name}",
                            lambda d, m: td.project(d, d.fields[name], td.Constant(d)))
        elif _kind(field) not in ("basis", "values-cells"):
            self.error(f"{fn}() takes cell data", pos)
        a = self.raw(f"{name}:cells", get)
        t = _type(field)
        return self.load(f"{fn}:{name}", f"{a}[{'s0' if fn == 'owner' else 's1'}]", t)

    def fsum(self, name, pos):
        """fsum(F): the outward sum of face values over each cell's faces."""
        if self.mode not in ("cells", "nodes"):
            self.error("fsum() is for cell expressions", pos)
        field = self.lookup(name, pos)
        if _kind(field) != "values-faces" or _is_vector(field):
            self.error("fsum() takes a scalar face field", pos)
        a = self.raw(name, self.by_name(name))
        k = len(self.lines)
        n = "dom.NUM_FACES" if self.reference else "dom.num_faces(c)"
        self.lines += [
            f"fs{k} = 0.0 * {a}[0]",
            f"for k{k} in range({n}):",
            f"    fs{k} += {a}[dom.face_id(c, k{k})] * (1.0 if dom.face_side(c, k{k}) == 0 else -1.0)",
        ]
        return self.load(f"fsum:{name}", f"fs{k}", S)

    def tensor_gradients(self, name, pos, getter=None):
        """The world gradients of a tensor field's nine entries, row by row, as codes."""
        getter = getter or self.by_name(name)
        parts = self.once(f"tensor-components:{name}", lambda d, m, g=getter: _tensor_components(g(d, m)))
        pc = self.pc(pos)
        self.uses_world_gradient = True
        out = []
        for k in range(9):
            u = self.arg("field", f"{name}#t{k}", lambda d, m, k=k: parts(d, m)[k])
            out.append(self.load(f"grad:{name}#t{k}", self.world_gradient(u, pc), V)[0])
        return out

    def gradient(self, fn, name, pos, field=None, getter=None):
        """grad(u) of a scalar; div, curl, qcrit of a vector, from its components' gradients."""
        if self.mode not in ("cells", "nodes"):
            self.error(f"{fn}() is for cell and node expressions; a point gradient needs "
                       "averaging over cells (td.gradients(at='points'))", pos)
        field = field if field is not None else self.lookup(name, pos)
        getter = getter or self.by_name(name)
        if _kind(field) == "hdiv" and fn == "div" and hasattr(td, "hdiv_div"):
            self.uses_hdiv_div = True
            u = self.arg("field", name, getter)
            return self.load(f"div:{name}", f"hdiv_div(dom, {u}, c, {self.pc(pos)})", S)
        if _kind(field) == "hdiv" and fn == "div":
            # Each face function's reference divergence is 1: div = sum of outward fluxes / det J.
            pc = self.pc(pos)
            a = self.raw(name, getter)
            k = len(self.lines)
            self.lines += [
                f"dv{k} = 0.0 * {a}[0]",
                f"for k{k} in range(dom.NUM_FACES):",
                f"    dv{k} += {a}[dom.face_id(c, k{k})] * (1.0 if dom.face_side(c, k{k}) == 0 else -1.0)",
            ]
            return self.load(f"div:{name}", f"dv{k} / dom.geometry_jacobian(c, {pc}).determinant()", S)
        if _kind(field) == "hcurl" and fn == "curl":
            self.uses_hcurl_curl = True
            u = self.arg("field", name, getter)
            return self.load(f"curl:{name}", f"hcurl_curl(dom, {u}, c, {self.pc(pos)})", V)
        if _kind(field) != "basis":
            self.error(f"{fn}() takes a field with a basis", pos)
        pc = self.pc(pos)
        self.uses_world_gradient = True
        if _type(field) == M:
            if fn != "div":
                self.error(f"{fn}() of a tensor field is not implemented; div() is, and grad() of each "
                           "entry", pos)
            g = self.tensor_gradients(name, pos, getter)
            rows = [" + ".join(f"{g[3 * i + j]}[{j}]" for j in range(3)) for i in range(3)]
            return self.load(f"div:{name}", f"tack.vec3({', '.join(rows)})", V)
        if fn == "grad" and not _is_vector(field):
            u = self.arg("field", name, getter)
            return self.load(f"grad:{name}", self.world_gradient(u, pc), V)
        if not _is_vector(field):
            self.error(f"{fn}() takes a vector field", pos)
        rows = []
        parts = self.once(f"components:{name}", lambda d, m, g=getter: _components(g(d, m)))
        for k in range(3):
            u = self.arg("field", f"{name}#{k}", lambda d, m, k=k: parts(d, m)[k])
            rows.append(self.load(f"grad:{name}#{k}", self.world_gradient(u, pc), V)[0])
        gx, gy, gz = rows   # row i: the gradient of component i
        if fn == "grad":
            return self.load(f"jac:{name}", f"tack.mat3({gx}, {gy}, {gz})", M)
        if fn == "div":
            return self.emit(f"{gx}[0] + {gy}[1] + {gz}[2]", S)
        if fn == "curl":
            return self.emit(f"tack.vec3({gz}[1] - {gy}[2], {gx}[2] - {gz}[0], {gy}[0] - {gx}[1])", V)
        return self.emit(f"-0.5 * ({gx}[0] * {gx}[0] + {gy}[1] * {gy}[1] + {gz}[2] * {gz}[2])"
                         f" - ({gx}[1] * {gy}[0] + {gx}[2] * {gz}[0] + {gy}[2] * {gz}[1])", S)

    # --- grammar --------------------------------------------------------

    def restart(self, text):
        """Parse another expression into the same kernel: its field reads, derivatives and
        constants are shared with the ones before it."""
        self.text = text
        self.toks = self._tokenize(text)
        self.k = 0

    def parse(self):
        result = self.expr(0)
        kind, val, pos = self.peek()
        if kind != "end":
            self.error(f"unexpected '{val}'", pos)
        return result

    def expr(self, rbp):
        left = self.prefix(self.take())
        while True:
            kind, val, _ = self.peek()
            bp = _INFIX[val][0] if kind == "op" and val in _INFIX else 0
            if rbp >= bp:
                return left
            left = self.infix(self.take(), left)

    def prefix(self, tok):
        kind, val, pos = tok
        if kind == "num":
            return self.const(val)
        if kind == "str":
            return self.field(val[1:-1], pos)
        if kind == "name":
            if self.peek()[1] == "(":
                return self.call(val, pos)
            return self.name(val, pos)
        if val == "(":
            result = self.expr(0)
            self.expect(")")
            return result
        if val == "{":
            parts = self.args_list("}")
            if len(parts) == 3 and all(t == V for _, t in parts):
                return self.emit(f"tack.mat3({', '.join(c for c, _ in parts)})", M)
            if len(parts) != 3 or any(t != S for _, t in parts):
                self.error("a vector needs three scalars, and a tensor three vectors (its rows)", pos)
            return self.emit(f"tack.vec3({', '.join(c for c, _ in parts)})", V)
        if val in ("-", "+", "!", "not"):
            a, t = self.expr(_PREFIX_BP)
            if val == "+":
                return a, t
            if val == "-":
                return self.emit(f"-{a}", t)
            self.need(t, S, pos)
            return self.emit(f"tack.select({a} == 0.0, 1.0, 0.0)", S)
        raise self.failure("the expression ends too early" if kind == "end" else f"unexpected '{val}'", pos)

    def name(self, val, pos):
        if self.has_field(val):
            return self.field(val, pos)
        if val in ("iHat", "jHat", "kHat"):
            j = "ijk".index(val[0])
            return "tack.vec3(" + ", ".join("1.0" if c == j else "0.0" for c in range(3)) + ")", V
        if val == "pi":
            return "3.141592653589793", S
        if val in ("true", "false"):
            return ("1.0" if val == "true" else "0.0"), S
        if val in ("coords", "coordsX", "coordsY", "coordsZ"):
            x = self.coords(pos)
            if val == "coords":
                return x
            return self.emit(f"{x[0]}[{'XYZ'.index(val[-1])}]", S)
        if val == "identity":
            return "tack.identity(3)", M
        base, _, comp = val.rpartition("_")
        if comp in ("X", "Y", "Z") and self.has_field(base):
            a, t = self.field(base, pos)
            self.need(t, V, pos)
            return self.emit(f"{a}[{'XYZ'.index(comp)}]", S)
        if len(comp) == 2 and set(comp) <= set("XYZ") and self.has_field(base):
            a, t = self.field(base, pos)
            self.need(t, M, pos)
            return self.emit(f"{a}[{'XYZ'.index(comp[0])}, {'XYZ'.index(comp[1])}]", S)
        raise self.failure(f"unknown name '{val}'", pos)

    def coords(self, pos):
        if self.mode == "faces":
            self.error("coords is for cell, point and node expressions", pos)
        if self.mode == "pointloop":
            geometry = self.data.geometry
            if not _entity_numbered_h1(geometry):
                raise _NeedCorners()
            return self.load("coords", f"{self.raw('geometry', lambda d, m: d.geometry)}[p]", V)
        if self.mode == "corners" and not self.reference:
            g = self.raw("geometry", lambda d, m: d.geometry)
            return self.load("coords", f"{g}[dom.point_id(c, j)]", V)
        return self.load("coords", f"dom.position(c, {self.pc(pos)})", V)

    def args_list(self, close):
        parts = []
        if self.peek()[1] != close:
            parts.append(self.expr(0))
            while self.peek()[1] == ",":
                self.take()
                parts.append(self.expr(0))
        self.expect(close)
        return parts

    def field_arg(self, fn):
        self.expect("(")
        kind, name, npos = self.take()
        if kind == "str":
            name = name[1:-1]
        elif kind != "name":
            self.error(f"{fn}() takes a field name", npos)
        self.expect(")")
        return name, npos

    def derived(self, fn, t, d, pos):
        """grad, div, curl or qcrit from an expression's derivative (None: zero)."""
        from ._derivative import ZERO_M, ZERO_V
        if fn == "grad":
            if t not in (S, V):
                self.error("grad() takes a scalar or vector expression", pos)
            return ((d or ZERO_V), V) if t == S else ((d or ZERO_M), M)
        if fn == "div" and t == M:
            if not d:
                return ZERO_V, V
            rows = [" + ".join(f"{d[m]}[{i}, {m}]" for m in range(3)) for i in range(3)]
            return self.emit(f"tack.vec3({', '.join(rows)})", V)
        if t != V:
            self.error(f"{fn}() takes a vector expression", pos)
        J = d or ZERO_M
        if fn == "div":
            return self.emit(f"{J}[0, 0] + {J}[1, 1] + {J}[2, 2]", S)
        if fn == "curl":
            return self.emit(f"tack.vec3({J}[2, 1] - {J}[1, 2], {J}[0, 2] - {J}[2, 0], {J}[1, 0] - {J}[0, 1])", V)
        return self.emit(f"-0.5 * ({J}[0, 0] * {J}[0, 0] + {J}[1, 1] * {J}[1, 1] + {J}[2, 2] * {J}[2, 2])"
                         f" - ({J}[0, 1] * {J}[1, 0] + {J}[0, 2] * {J}[2, 0] + {J}[1, 2] * {J}[2, 1])", S)

    def call(self, fn, pos):
        if fn == "laplacian":
            from ._second import argument_tokens, hessian_of
            inner, self.k = argument_tokens(self.toks, self.k, self.error, pos, fn)
            if self.mode not in ("cells", "nodes"):
                self.error("laplacian() is for cell and node expressions", pos)
            _, _, H = hessian_of(self, inner, pos)
            return self.emit(f"{H}[0, 0] + {H}[1, 1] + {H}[2, 2]", S) if H else ("0.0", S)
        if fn in _FIELD_FUNCTIONS:
            simple = (len(self.toks) > self.k + 2 and self.toks[self.k][1] == "("
                      and self.toks[self.k + 1][0] in ("name", "str")
                      and self.toks[self.k + 2][1] == ")")
            if not simple and fn in ("grad", "div", "curl", "qcrit"):
                from ._derivative import compile_argument
                _, t, d = compile_argument(self, fn, pos)
                return self.derived(fn, t, d, pos)
            name, npos = self.field_arg(fn)
            if fn in ("grad", "div", "curl", "qcrit"):
                return self.gradient(fn, name, npos)
            if fn == "fsum":
                return self.fsum(name, npos)
            read = {"points": self.points_of, "cells": self.cells_of}.get(fn)
            got = read(name, npos) if read else self.side(fn, name, npos)
            return self.matrix(self.lookup(name, npos), got, f"{fn}:{name}")
        self.expect("(")
        a = self.args_list(")")
        c = [code for code, _ in a]
        t = [typ for _, typ in a]
        got = self.tensor_call(fn, c, t, pos)
        if got is not None:
            return got

        def check(*want):
            if len(a) != len(want):
                self.error(f"{fn}() takes {len(want)} argument{'s' if len(want) > 1 else ''}", pos)
            for got, w in zip(t, want):
                self.need(got, w, pos)

        if fn in _UNARY_S:
            check(S)
            return self.emit(f"{_UNARY_S[fn]}({c[0]})", S)
        if fn in ("min", "max", "atan2"):
            check(S, S)
            return self.emit(f"{fn}({c[0]}, {c[1]})", S)
        if fn == "pow":
            check(S, S)
            return self.emit(f"{c[0]} ** {c[1]}", S)
        if fn == "mag":
            check(V)
            return self.emit(f"{c[0]}.norm()", S)
        if fn == "mag2":
            check(V)
            return self.emit(f"{c[0]}.norm_sqr()", S)
        if fn == "norm":
            check(V)
            return self.emit(f"{c[0]}.normalized()", V)
        if fn == "dot":
            check(V, V)
            return self.emit(f"{c[0]}.dot({c[1]})", S)
        if fn == "cross":
            check(V, V)
            return self.emit(f"{c[0]}.cross({c[1]})", V)
        if fn == "if":
            if len(a) != 3:
                self.error("if() takes 3 arguments", pos)
            return self.select(a[0], a[1], a[2], pos)
        if fn == "clamp":
            check(S, S, S)
            return self.emit(f"min(max({c[1]}, {c[0]}), {c[2]})", S)
        if fn == "not":
            check(S)
            return self.emit(f"tack.select({c[0]} == 0.0, 1.0, 0.0)", S)
        raise self.failure(f"unknown function '{fn}'", pos)

    def eigvals(self, a):
        """The eigenvalues of a's symmetric part, largest first (Smith's closed form)."""
        k = len(self.lines)
        self.lines += [
            f"es{k} = 0.5 * ({a} + {a}.transpose())",
            f"eq{k} = es{k}.trace() / 3.0",
            f"ep1{k} = es{k}[0, 1] * es{k}[0, 1] + es{k}[0, 2] * es{k}[0, 2] + es{k}[1, 2] * es{k}[1, 2]",
            (f"ep2{k} = (es{k}[0, 0] - eq{k}) * (es{k}[0, 0] - eq{k}) + (es{k}[1, 1] - eq{k}) * (es{k}[1, 1] - eq{k})"
             f" + (es{k}[2, 2] - eq{k}) * (es{k}[2, 2] - eq{k}) + 2.0 * ep1{k}"),
            f"ep{k} = sqrt(ep2{k} / 6.0)",
            f"eb{k} = (es{k} - eq{k} * tack.identity(3)) / max(ep{k}, 1e-30)",
            f"er{k} = min(max(0.5 * eb{k}.determinant(), -1.0), 1.0)",
            f"ephi{k} = acos(er{k}) / 3.0",
            f"ea{k} = eq{k} + 2.0 * ep{k} * cos(ephi{k})",
            f"ec{k} = eq{k} + 2.0 * ep{k} * cos(ephi{k} + 2.0943951023931953)",
        ]
        return self.emit(f"tack.vec3(ea{k}, 3.0 * eq{k} - ea{k} - ec{k}, ec{k})", V)

    def tensor_call(self, fn, c, t, pos):
        """The tensor functions, or None for another function."""
        def want(*types):
            if tuple(t) != types:
                names = ", ".join(types)
                self.error(f"{fn}() takes ({names}), got ({', '.join(t)})", pos)

        if fn in ("trace", "det", "transpose", "inv", "sym", "skew", "dev", "vonmises", "eigvals"):
            want(M)
            a = c[0]
            if fn == "trace":
                return self.emit(f"{a}.trace()", S)
            if fn == "det":
                return self.emit(f"{a}.determinant()", S)
            if fn == "transpose":
                return self.emit(f"{a}.transpose()", M)
            if fn == "inv":
                return self.emit(f"{a}.inverse()", M)
            if fn == "sym":
                return self.emit(f"0.5 * ({a} + {a}.transpose())", M)
            if fn == "skew":
                return self.emit(f"0.5 * ({a} - {a}.transpose())", M)
            dev, _ = self.emit(f"{a} - ({a}.trace() / 3.0) * tack.identity(3)", M)
            if fn == "dev":
                return dev, M
            if fn == "vonmises":
                return self.emit(f"sqrt(1.5 * ({dev}.transpose() @ {dev}).trace())", S)
            return self.eigvals(a)
        if fn == "outer":
            want(V, V)
            return self.emit(f"{c[0]}.outer_product({c[1]})", M)
        if fn == "ddot":
            want(M, M)
            return self.emit(f"({c[0]}.transpose() @ {c[1]}).trace()", S)
        if fn in ("mag", "mag2") and t == [M]:
            m2 = f"({c[0]}.transpose() @ {c[0]}).trace()"
            return self.emit(f"sqrt({m2})" if fn == "mag" else m2, S)
        if fn == "dot" and M in t and len(t) == 2:
            if t[0] == S or t[1] == S:
                self.error("dot() takes vectors and tensors; multiply by a scalar with '*'", pos)
            return self.emit(f"{c[0]} @ {c[1]}", M if t == [M, M] else V)
        return None

    def infix(self, tok, left):
        _, op, pos = tok
        bp, right_assoc = _INFIX[op]
        if op == "?":
            a = self.expr(0)
            self.expect(":")
            b = self.expr(bp - 1)
            return self.select(left, a, b, pos)
        if op == "[":
            idx, _ = self.expr(0)
            self.expect("]")
            if left[1] == S:
                self.error("an index takes a vector's component or a tensor's row", pos)
            k = self.const_value(idx)
            if k not in (0.0, 1.0, 2.0):
                self.error("an index must be 0, 1 or 2", pos)
            if left[1] == M:
                k = int(k)
                return self.emit(f"tack.vec3({left[0]}[{k}, 0], {left[0]}[{k}, 1], {left[0]}[{k}, 2])", V)
            return self.emit(f"{left[0]}[{int(k)}]", S)
        (lc, lt), (rc, rt) = left, self.expr(bp - 1 if right_assoc else bp)
        if op in ("+", "-", "*", "/"):
            if M in (lt, rt):
                if op in ("+", "-") and lt != rt:
                    self.error(f"a tensor {'adds to' if op == '+' else 'subtracts'} only a tensor", pos)
                if op == "*" and S not in (lt, rt):
                    self.error("'*' multiplies a tensor by a scalar; use dot() for products of tensors and vectors",
                               pos)
                if op == "/" and rt != S:
                    self.error("a tensor divides only by a scalar", pos)
                return self.emit(f"{lc} {op} {rc}", M)
            if op == "/" and rt == V and lt == S:
                self.error("cannot divide a scalar by a vector", pos)
            return self.emit(f"{lc} {op} {rc}", V if V in (lt, rt) else S)
        self.need(lt, S, pos)
        self.need(rt, S, pos)
        if op == "^":
            return self.emit(f"{lc} ** {rc}", S)
        if op == "%":
            q, _ = self.emit(f"{lc} / {rc}", S)
            return self.emit(f"{lc} - {rc} * (floor({q}) if {q} >= 0.0 else ceil({q}))", S)
        if op in _COMPARE:
            return self.emit(f"tack.select({lc} {_COMPARE[op]} {rc}, 1.0, 0.0)", S)
        if op in ("and", "&&", "&"):
            return self.emit(f"tack.select({lc} != 0.0, tack.select({rc} != 0.0, 1.0, 0.0), 0.0)", S)
        return self.emit(f"tack.select({lc} != 0.0, 1.0, tack.select({rc} != 0.0, 1.0, 0.0))", S)

    def select(self, c, a, b, pos):
        self.need(c[1], S, pos)
        if a[1] != b[1]:
            self.error("both branches must have the same type", pos)
        return self.emit(f"tack.select({c[0]} != 0.0, {a[0]}, {b[0]})", a[1])


# --- pre-passes shared across expressions ---------------------------------

_links_cache = {}


def _components(field):
    """A vector field's three components as scalar fields in its space, copied now, each
    with its row of the field's given gradient where it has one."""
    values = materialize(field.values)
    rows = _gradient_rows(getattr(field, "gradient", None))
    parts = []
    for k in range(3):
        part = tack.empty((field.space.size,), tack.arrays.dtype_of(values))
        component(values, k, part, field.space.size)
        if rows is None:
            parts.append(td.Field(field.space, part))
        else:
            parts.append(td.Field(field.space, part, gradient=rows[k]))
    return parts


def _gradient_rows(given):
    """A vector field's given gradient (9 values each, row by row) as three fields of rows."""
    if given is None:
        return None
    values = materialize(given.values)
    flat = np.asarray(values.numpy()).reshape(given.space.size, 9)
    out = []
    for k in range(3):
        row = tack.empty((given.space.size,), tack.vec3(values.scalar_dtype))
        tack.copyto(row, np.ascontiguousarray(flat[:, 3 * k:3 * k + 3]))
        out.append(td.Field(given.space, row, given.unit))
    return out


def _stored_expression(field):
    """The expression field a name refers to, or None."""
    space = getattr(field, "space", None)
    return field if type(space).__name__ == "ExpressionSpace" and hasattr(space, "text") else None


_TAKES_A_NAME = ("points", "cells", "owner", "neighbor", "fsum")


def _tensor_components(field):
    """A tensor field's nine entries, row by row, as scalar fields in its space, copied now."""
    values = materialize(field.values)
    scalar = values.scalar_dtype
    flat = np.asarray(values.numpy(), dtype=np.float64).reshape(field.space.size, -1)
    if flat.shape[1] == 6:   # symmetric: XX YY ZZ XY YZ XZ
        xx, yy, zz, xy, yz, xz = flat.T
        flat = np.stack([xx, xy, xz, xy, yy, yz, xz, yz, zz], axis=1)
    parts = []
    for k in range(9):
        part = tack.empty((field.space.size,), scalar)
        tack.copyto(part, np.ascontiguousarray(flat[:, k]))
        parts.append(td.Field(field.space, part))
    return parts


def _grid_dims(data):
    """[nx, ny, nz, cx, cy, cz] (points and cells per axis) for a structured grid whose
    points and cells are numbered x fastest on its own axes, else None. A grid with a
    size of 1 before a larger axis has its axes renumbered, and goes through td.project."""
    topology = data.topology
    if not isinstance(topology, td.StructuredTopology):
        return None
    dims = list(getattr(topology, "point_dims", ()))
    if not dims or len(dims) > 3:
        return None
    dims += [1] * (3 - len(dims))
    if any(dims[a] == 1 and any(d > 1 for d in dims[a + 1:]) for a in range(3)):
        return None
    key = ("grid", id(topology))
    if key not in _links_cache:
        cells = [max(d - 1, 1) for d in dims]
        _links_cache[key] = (topology, tack.array(np.array(dims + cells, dtype=np.int32)))
    return _links_cache[key][1]


def _point_links(data):
    """(offsets, cell of each link) for an unstructured mesh of reference cells, one shape
    or several, else None (polyhedra)."""
    from tack.dataset.topology import UnstructuredTopology, corner_layout
    topology = data.topology
    if isinstance(topology, td.PolyhedralTopology):
        return _polyhedral_links(topology)
    if not isinstance(topology, UnstructuredTopology) or not getattr(topology, "reference_cells", True):
        return None
    key = ("links", id(topology))
    if key not in _links_cache:
        offsets, entries = topology.point_links()
        start_of, total = corner_layout(topology)
        n = entries.shape[0]
        cells = tack.empty((n,), tack.i32)
        for group in topology.groups():
            if group.count and n:
                start = start_of[group]
                view = group.view()
                entry_cells(view, entries, cells, n, start, start + group.count * group.shape.NUM_POINTS)
        _links_cache[key] = (topology, (offsets, cells))
    return _links_cache[key][1]


def _polyhedral_links(topology):
    """(offsets, cell of each link) for polyhedra or polygons: point_links() indexes each
    cell's point list in cell_points(), so each list position is mapped to its cell."""
    key = ("links", id(topology))
    if key not in _links_cache:
        offsets, entries = topology.point_links()
        point_offsets, point_ids = topology.cell_points()
        cell_of_position = tack.empty((point_ids.shape[0],), tack.i32)
        position_cells(point_offsets, cell_of_position, topology.num_cells)
        cells = tack.empty((entries.shape[0],), tack.i32)
        if entries.shape[0]:
            gather(cell_of_position, entries, cells, entries.shape[0])
        _links_cache[key] = (topology, (offsets, cells))
    return _links_cache[key][1]


# --- kernel assembly ------------------------------------------------------

_ASSIGNMENT = re.compile(r"([A-Za-z_]\w*) = .*")
_NAME = re.compile(r"[A-Za-z_]\w*")


def _live(lines, results):
    """lines without the top-level assignments that nothing after them reads, results
    included. A pass computes every component, gradient and Hessian of what it reads, and
    a kernel pays to compile all of them; loops and other lines stay, with what they read."""
    needed = set(_NAME.findall(" ".join(results)))
    kept = []
    for line in reversed(lines):
        m = _ASSIGNMENT.fullmatch(line)
        if m and m.group(1) not in needed:
            continue
        kept.append(line)
        needed.update(_NAME.findall(line))
    return kept[::-1]


def _loop(mode, body, results, uses_pc, count="(target.ORDER + 1) ** dom.DIMENSION"):
    """The kernel's loop: body, then each result stored in its output (out, or out0, out1...)."""
    outs = ["out"] if len(results) == 1 else [f"out{k}" for k in range(len(results))]

    def store(indent, index):
        return [f"{indent}{o}[{index}] = {r}" for o, r in zip(outs, results)]

    if mode == "cells":
        head = ["for c in dom:"] + (["    pc = dom.parametric_center()"] if uses_pc else [])
        return head + ["    " + b for b in body] + store("    ", "dom.entity_id(c)")
    if mode == "corners":
        head = ["for c in dom:", "    for j in range(" + ("dom.NUM_POINTS" if count else "dom.num_points(c)") + "):"]
        head += ["        pc = dom.parametric_point(j)"] if uses_pc else []
        return head + ["        " + b for b in body] + store("        ", "dom.point_id(c, j)")
    if mode == "nodes":
        count, pc, index = _NODE_LOOPS[count]
        head = ["for c in dom:", f"    for n in range({count}):", f"        pc = {pc}"]
        return head + ["        " + b for b in body] + store("        ", index)
    if mode == "pointloop":
        return ["for p in range(n):"] + ["    " + b for b in body] + store("    ", "p")
    head = ["for f in dom:", "    e = dom.entity_id(f)", "    s0 = dom.side_cell(f, 0)",
            "    s1 = dom.side_cell(f, 1)", "    s1 = s1 if s1 >= 0 else s0"]
    return head + ["    " + b for b in body] + store("    ", "e")


# Node loops by the target space and cell shape: (count, parametric point, value index).
_NODE_LOOPS = {
    "tensor": ("(target.ORDER + 1) ** dom.DIMENSION", "target.node_parametric(n)", "target.dof_index(c, n)"),
    "lattice": ("target.NODES", "target.node_parametric(n)", "target.dof_index(c, n)"),
    "corners": ("dom.NUM_POINTS", "dom.parametric_point(n)", "target.dof_index(c, n)"),
    "center": ("1", "dom.parametric_center()", "target.dof_index(c, 0)"),
    "listed": ("node_count", "target.node_parametric(node_list[n])", "target.dof_index(c, node_list[n])"),
    "quadratic": ("node_count", "dom.quadratic_node(n)", "target.dof_index(c, n)"),
}


def _node_variant(space, shape):
    """Which node loop fills space on cells of shape, and its node list or count."""
    from tack.dataset import bases
    if isinstance(space, td.H1):
        any_order = getattr(space, "reads_any_order", None) or space._any_order
        if any_order(shape):
            return "listed", space.stored_nodes(shape)
        if hasattr(space, "quadratic_count"):
            return "quadratic", space.quadratic_count(shape)
        from tack.dataset.algorithms.project import (
            _quadratic_count,  # a Tack without the public one
        )
        return "quadratic", _quadratic_count(space, shape)
    if not space.tensor:
        return ("center" if space.order == 0 else "corners"), None
    return ("tensor" if bases.tensor_shape(shape) else "lattice"), None


_compiled = {}


def _load(source, name):
    key = hashlib.sha1(source.encode()).hexdigest()[:16]
    if key not in _compiled:
        os.makedirs(CACHE_DIR, exist_ok=True)
        path = os.path.join(CACHE_DIR, f"fx_{key}.py")
        if not os.path.exists(path):
            with open(path, "w") as f:
                f.write(source)
        else:
            try:
                os.utime(path)          # marks it used, for clear_cache's age
            except OSError:
                pass
        spec = importlib.util.spec_from_file_location(f"tack_expr_fx_{key}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _compiled[key] = getattr(module, name)
    return _compiled[key]


def _imports(c):
    imports = "import tack\nfrom tack import math as tm\n"
    if c.uses_vector_spaces:
        imports += "from tack.dataset import hcurl_value, hdiv_value\n"
    if c.uses_derivatives:
        imports += "from tack_expr import _ad as ad\n"
    if c.uses_hcurl_curl:
        imports += "from tack.dataset import hcurl_curl\n"
    if c.uses_hdiv_div:
        imports += "from tack.dataset import hdiv_div\n"
    return imports + (_GRADIENT_IMPORT if c.uses_world_gradient else "")


def _scalar(data, value):
    return np.float64(value) if data.dtype is tack.f64 else np.float32(value)


def _incidence(code):
    """The cell incidence generated code reads through its domain view (``faces``,
    ``edges``), which ``domain_view`` and ``td.for_each`` give a view only when asked."""
    return tuple(kind for kind, read in (("faces", "dom.face_"), ("edges", "dom.edge_"))
                 if read in code)


class Compiled:
    """Compiled expressions: their kernels, output location and arguments.

    Kernel sources depend only on the expressions' structure, the output and how
    each field is read. Constants are scalar arguments, so expressions that differ
    only in their numbers share compiled kernels. compile_expression makes one with a
    single output; compile_many one with several, computed in the same loop.
    """

    def __init__(self, text, make, mode, space, args, consts, types, names=None):
        self.text, self.make, self.mode, self.space = text, make, mode, space
        self.args, self.consts, self.types, self.names = args, consts, types, names
        self.sources, self.kernels = {}, {}
        self.unit = None
        self.incidence = ()

    @property
    def type(self):
        return self.types[0]

    def kernel(self, variant):
        if variant not in self.kernels:
            self.sources[variant], self.kernels[variant] = self.make(variant)
        return self.kernels[variant]

    @property
    def source(self):
        if not self.sources:
            self.kernel("main" if self.mode != "nodes" else "tensor")
        return next(iter(self.sources.values()))

    def _allocate(self, data, t):
        dtype = tack.vec3(data.dtype) if t == V else tack.mat3(data.dtype) if t == M else data.dtype
        # A point no cell uses is not written in the corner loop.
        alloc = tack.zeros if self.mode == "corners" else tack.empty
        return alloc((self.space.size,), dtype)

    def __call__(self, data, out=None):
        """Run over data. With one expression: a td.Field, out its array if given. With
        several: {name: td.Field}, out a {name: array} for any of them to fill."""
        if self.names is None:
            outs = [out if out is not None else self._allocate(data, self.types[0])]
        else:
            given = out or {}
            outs = [given.get(n) if given.get(n) is not None else self._allocate(data, t)
                    for n, t in zip(self.names, self.types)]
        unit = self.unit if isinstance(self.unit, dict) else {None: self.unit}
        results = [td.Field(self.space, o, unit.get(n)) for n, o in zip(self.names or [None], outs)]
        memo = {}
        payloads = [get(data, memo) for _, get in self.args]
        consts = [_scalar(data, v) for v in self.consts]
        if self.mode == "pointloop":
            self.kernel("main")(data.num_points, *payloads, *outs, *consts)
        elif self.mode == "nodes":
            kinds = [kind for kind, _ in self.args]
            fields = [p for kind, p in zip(kinds, payloads) if kind == "field"]
            for group in data.launch_groups("cells", [*fields, results[0]]):
                if not group.count:
                    continue
                variant, nodes = _node_variant(self.space, group.shape)
                node_list = tack.array(np.asarray(nodes if hasattr(nodes, "__len__") else [0], dtype=np.int32))
                node_count = len(nodes) if hasattr(nodes, "__len__") else int(nodes or 0)
                self.kernel(variant)(data.domain_view("cells", group, incidence=self.incidence),
                                     *[p.view(group) if kind == "field" else p
                                       for kind, p in zip(kinds, payloads)],
                                     results[0].view(group), node_list, node_count, *outs, *consts)
        else:
            domain = "faces" if self.mode == "faces" else "cells"
            td.for_each(self.kernel("main"), data, domain, *payloads, *outs, *consts,
                        incidence=self.incidence if domain == "cells" else ())
        return results[0] if self.names is None else dict(zip(self.names, results))


def _output(data, at):
    if isinstance(at, str):
        if at == "cells":
            return "cells", td.Constant(data)
        if at == "points":
            return "pointloop", td.H1(data)
        if at == "faces":
            return "faces", td.Values(data, "faces")
        raise ValueError(f"at must be 'cells', 'points', 'faces' or a space, not {at!r}")
    if isinstance(at, td.H1) and at.order == 1 and at.topology is data.topology:
        return "pointloop", at
    if isinstance(at, (td.L2, td.H1)) and at.topology is data.topology:
        return "nodes", at
    raise ValueError("node output takes an L2 or H1 space of the dataset's topology")


def _compile(texts, data, at, replace_invalid, names):
    mode, space = _output(data, at)
    continuous = isinstance(space, td.H1) and mode == "nodes"

    def parse_all(mode, continuous):
        c = _Compiler(texts[0], data, mode, continuous)
        parsed = [c.parse()]
        for text in texts[1:]:
            c.restart(text)
            parsed.append(c.parse())
        return c, parsed

    try:
        c, parsed = parse_all(mode, continuous)
    except _NeedCorners:
        mode = "corners"
        c, parsed = parse_all(mode, False)
    results, types = [], []
    for result, rtype in parsed:
        if replace_invalid is not None and rtype != M:
            rep, _ = c.const(replace_invalid)
            result, _ = c.emit(f"tack.select({result} - {result} == 0.0, {result}, {rep})", rtype)
        results.append(result)
        types.append(rtype)
    outs = ["out"] if len(results) == 1 else [f"out{k}" for k in range(len(results))]
    params = (["n"] if mode == "pointloop" else ["dom"]) + [f"A{k}" for k in range(len(c.args))]
    params += (["target", "node_list", "node_count"] if mode == "nodes" else [])
    params += outs + [f"c{k}" for k in range(len(c.consts))]
    head = _imports(c) + "\n\n@tack.kernel\ndef fx(" + ", ".join(params) + "):\n"

    lines, uses_pc, reference = _live(c.lines, results), c.uses_pc, c.reference

    def make(variant):
        count = variant if mode == "nodes" else ("dom.NUM_POINTS" if reference else "")
        source = head + "\n".join("    " + line for line in _loop(mode, lines, results, uses_pc, count)) + "\n"
        return source, _load(source, "fx")

    text = texts[0] if names is None else "; ".join(f"{n}={t}" for n, t in zip(names, texts))
    compiled = Compiled(text, make, mode, space, list(c.args), list(c.consts), types, names)
    compiled.fields = sorted(c.fields_read)
    compiled.incidence = _incidence("\n".join(lines))
    return compiled


def _prepared(sources, data, at, replace_invalid, names, fields, units):
    """Compile through the tree passes of _front: names bound by the field dictionary, complex
    values split, units checked and converted."""
    from . import _front
    labels, texts, result_units = {}, [], []
    for k, source in enumerate(sources):
        target = units.get(names[k]) if isinstance(units, dict) else units
        text, labels, unit = _front.prepare(source, data, fields, target, labels)
        texts.append(text)
        result_units.append(unit)
    with bound({**(_BINDINGS.get() or {}), **labels}):
        compiled = _compile(texts, data, at, replace_invalid, names)
    compiled.unit = result_units[0] if names is None else dict(zip(names, result_units))
    return compiled


def compile_expression(text, data, at="cells", replace_invalid=None, fields=None, units=None):
    """Parse text over data's fields and return a Compiled expression for output at.

    replace_invalid: a number that replaces NaN and infinite results, as
    vtkArrayCalculator's ReplaceInvalidValues does.

    fields: a tack_expr.FieldDictionary, or {name: unit}, giving the names the expression
    may use and the units of fields that have none of their own (Field.unit). Units are
    checked and converted where they meet, and with units ("kPa") the result is given in
    them. compiled.unit is the result's unit, None when unknown."""
    from ._front import expands, needs_front
    if needs_front(text, fields, units, data):
        return _prepared([text], data, at, replace_invalid, None, fields, units)
    try:
        return _compile([text], data, at, replace_invalid, None)
    except ExprError:
        if expands(text, data):   # derivatives of tensors, written out by component
            return _prepared([text], data, at, replace_invalid, None, fields, units)
        raise


def compile_many(expressions, data, at="cells", replace_invalid=None, fields=None, units=None):
    """Several expressions, {name: text}, compiled into one kernel for output at: one
    launch, and each field read, derivative and conversion done once for all of them.
    Calling the result gives {name: td.Field}. units: one unit for all, or {name: unit}."""
    from ._front import needs_front
    if not expressions:
        raise ValueError("compile_many needs at least one expression")
    names = list(expressions)
    sources = [expressions[n] for n in names]
    if any(needs_front(s, fields, units, data) for s in sources):
        return _prepared(sources, data, at, replace_invalid, names, fields, units)
    try:
        return _compile(sources, data, at, replace_invalid, names)
    except ExprError:
        if any(expands(s, data) for s in sources):
            return _prepared(sources, data, at, replace_invalid, names, fields, units)
        raise


def _exchangeable(field):
    """A field td.update_ghosts refreshes: explicit values on points or cells."""
    space = field.space
    on_points = (isinstance(space, td.H1) and space.order == 1) or (isinstance(space, td.Values) and space.on == "points")
    on_cells = isinstance(space, td.Constant) or (isinstance(space, td.Values) and space.on == "cells")
    return (on_points or on_cells) and isinstance(field.values, tack.Array)


def evaluate(text, data, at="cells", replace_invalid=None, name=None, update_ghosts=False, project=False,
             fields=None, units=None):
    """Evaluate text over data's fields at 'cells', 'points', 'faces' or a space.

    Returns a td.Field in the output space; with name, also stores it in data.fields.

    data may be a td.PartitionedDataSet: the expression runs on each piece, at may
    then be a function of the piece giving its space, and a list of fields comes
    back. With a ghost layer around each piece, the values each piece owns are the
    whole mesh's, conversions and face operators at the cuts included. For pieces
    made by td.ghost_cells or td.ghost_points, update_ghosts=True first refreshes the
    ghost values of the point and cell fields the expression reads, from the pieces
    that own them (across MPI ranks as well), so a chain of evaluations stays right
    on the ghost layer too.

    project=True with an H1 output evaluates into L2 of the same order and averages
    into H1 with td.project, for expressions of fields that are not continuous.

    fields and units: as compile_expression's; the field's unit is the result's."""
    if isinstance(data, td.PartitionedDataSet):
        pieces = list(data)
        if update_ghosts and pieces:
            first = compile_expression(text, pieces[0], at(pieces[0]) if callable(at) else at, replace_invalid,
                                       fields, units)
            names = [n for n in first.fields if _exchangeable(pieces[0].fields[n])]
            if names:
                td.update_ghosts(data, *names)
        return [evaluate(text, piece, at(piece) if callable(at) else at, replace_invalid, name,
                         project=project, fields=fields, units=units) for piece in pieces]
    if project and isinstance(at, td.H1):
        try:
            staging = td.L2(data, order=max(at.order, 1))
            compiled = compile_expression(text, data, staging, replace_invalid, fields, units)
        except (TypeError, ValueError, NotImplementedError) as error:
            if isinstance(error, ExprError):
                raise
            staging = td.L2(data, order=max(at.order, 1), basis="equispaced")
            compiled = compile_expression(text, data, staging, replace_invalid, fields, units)
        field = td.project(data, compiled(data), at)
    else:
        compiled = compile_expression(text, data, at, replace_invalid, fields, units)
        field = compiled(data)
    if name is not None:
        data.fields[name] = field
    return field


def evaluate_complex(source, data, at="cells", replace_invalid=None, fields=None, units=None, project=False):
    """The real and imaginary parts of a complex expression, (real, imag) td.Fields, from one
    kernel. source is text or an operator expression; the other arguments are evaluate's."""
    if isinstance(source, str):
        parts = {"real": f"real({source})", "imag": f"imag({source})"}
    else:
        from . import ops
        parts = {"real": ops._Call("real", source), "imag": ops._Call("imag", source)}
    staging = td.L2(data, order=max(at.order, 1)) if project and isinstance(at, td.H1) else at
    out = compile_many(parts, data, staging, replace_invalid, fields, units)(data)
    if staging is not at:
        out = {k: td.project(data, f, at) for k, f in out.items()}
    return out["real"], out["imag"]


# --- expressions inside other kernels -------------------------------------

class Function:
    """An expression as a device class for use in a kernel: value(dom, c, pc) for one read
    at cells, value(dom, f) for one read at faces.

    Run kernels with tack_expr.for_each, which builds the object for each launch
    group: kernel(domain_view, expression_object, *args)."""

    def __init__(self, text, cls, args, consts, rtype, source, domain="cells"):
        self.text, self.cls, self.args = text, cls, args
        self.consts, self.type, self.source, self.domain = consts, rtype, source, domain
        self.incidence = _incidence(source)

    def resolve(self, data):
        memo = {}
        return [get(data, memo) for _, get in self.args]

    def bind(self, data, group, payloads):
        views = [p.view(group) if kind == "field" else p for (kind, _), p in zip(self.args, payloads)]
        # A device class takes Python numbers as instance scalars, not NumPy ones.
        return self.cls(*views, *[float(_scalar(data, v)) for v in self.consts])


def function(text, data, at="cells"):
    """Compile text, read at cells or at faces, as a Function for use inside other kernels."""
    if at not in ("cells", "faces"):
        raise ValueError("a function is read at 'cells' or 'faces'")
    c = _Compiler(text, data, at)
    result, rtype = c.parse()
    names = [f"A{k}" for k in range(len(c.args))] + [f"c{k}" for k in range(len(c.consts))]
    swap = re.compile(r"\b([Ac]\d+)\b")
    body = [swap.sub(r"self.\1", line) for line in _live(c.lines, [result])] + [f"return {swap.sub(r'self.' + chr(92) + '1', result)}"]
    if at == "faces":
        signature = "value(self, dom, f)"
        body = ["e = dom.entity_id(f)", "s0 = dom.side_cell(f, 0)", "s1 = dom.side_cell(f, 1)",
                "s1 = s1 if s1 >= 0 else s0", *body]
    else:
        signature = "value(self, dom, c, pc)"
    source = (_imports(c) + "\n\n@tack.device_class\nclass Fx:\n"
              "    def __init__(self" + "".join(f", {n}" for n in names) + "):\n"
              + "".join(f"        self.{n} = {n}\n" for n in names) + ("        pass\n" if not names else "")
              + f"\n    @tack.func\n    def {signature}:\n"
              + "".join(f"        {line}\n" for line in body))
    return Function(text, _load(source, "Fx"), list(c.args), list(c.consts), rtype, source, at)


def for_each(kernel, data, fn, *args):
    """Run kernel(domain_view, fn bound to the group, *args) once per group of the domain
    fn is read on: cells, or faces."""
    payloads = fn.resolve(data)
    fields = [p for (kind, _), p in zip(fn.args, payloads) if kind == "field"]
    fields += [a for a in args if isinstance(a, td.Field)]
    for group in data.launch_groups(fn.domain, fields):
        if group.count:
            kernel(data.domain_view(fn.domain, group,
                                    incidence=fn.incidence if fn.domain == "cells" else ()),
                   fn.bind(data, group, payloads),
                   *[a.view(group) if isinstance(a, td.Field) else a for a in args])


# --- the cache directory --------------------------------------------------

def clear_cache(older_than_days=None):
    """Remove generated modules from the cache directory; returns how many.

    older_than_days keeps modules used more recently than that. Tack's own
    kernel cache is separate and untouched."""
    if not os.path.isdir(CACHE_DIR):
        return 0
    cutoff = None if older_than_days is None else time.time() - 86400 * older_than_days
    removed = 0
    for name in os.listdir(CACHE_DIR):
        path = os.path.join(CACHE_DIR, name)
        if name.startswith("fx_") and name.endswith(".py"):
            if cutoff is None or os.path.getmtime(path) < cutoff:
                os.remove(path)
                removed += 1
    return removed


if CACHE_DAYS > 0:
    try:
        clear_cache(older_than_days=CACHE_DAYS)
    except OSError:
        pass

