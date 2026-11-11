# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Second derivatives: each scalar carries its value, gradient and Hessian through the
kernel; a vector is carried as its three components. grad() of an expression inside a
derivative uses this pass, so its Jacobian is the expression's Hessian, and laplacian()
is the Hessian's trace.

A scalar is (value, gradient, hessian), with None for a zero gradient or Hessian.
"""

from . import _ad as ad
from . import parser as P

S, V = P.S, P.V
ZV = "tack.vec3(0.0, 0.0, 0.0)"


class Vec:
    """A vector as three scalars, with its value."""

    def __init__(self, value, comps):
        self.value, self.comps = value, comps


class Second:
    def __init__(self, parent, toks):
        self.p = parent
        self.toks = toks
        self.k = 0
        parent.uses_derivatives = True

    # --- emission ---------------------------------------------------------

    def e(self, code):
        return self.p.emit(code, S)[0]

    def total(self, terms):
        terms = [t for t in terms if t is not None]
        return self.e(" + ".join(f"({t})" for t in terms)) if terms else None

    def const(self, value):
        return (value, None, None)

    def add(self, a, b, sign="+"):
        g = self.total([a[1], f"-({b[1]})" if (b[1] and sign == "-") else b[1]])
        H = self.total([a[2], f"-({b[2]})" if (b[2] and sign == "-") else b[2]])
        return (self.e(f"{a[0]} {sign} {b[0]}"), g, H)

    def mul(self, a, b):
        g = self.total([f"{a[0]} * {b[1]}" if b[1] else None, f"{b[0]} * {a[1]}" if a[1] else None])
        H = self.total([f"{a[0]} * {b[2]}" if b[2] else None, f"{b[0]} * {a[2]}" if a[2] else None,
                        f"{a[1]}.outer_product({b[1]}) + {b[1]}.outer_product({a[1]})" if (a[1] and b[1]) else None])
        return (self.e(f"{a[0]} * {b[0]}"), g, H)

    def chain(self, value, a, d1, d2):
        """f(a) with f' = d1 and f'' = d2 (code, or None for zero)."""
        if a[1] is None:
            return (value, None, None)
        g = self.e(f"({d1}) * {a[1]}") if d1 else None
        H = self.total([f"({d1}) * {a[2]}" if (d1 and a[2]) else None,
                        f"({d2}) * {a[1]}.outer_product({a[1]})" if d2 else None])
        return (value, g, H)

    def recip(self, b):
        x = b[0]
        v = self.e(f"1.0 / {x}")
        return self.chain(v, b, f"-{v} * {v}", f"2.0 * {v} * {v} * {v}")

    def blend(self, m, a, b):
        """select with a 0/1 mask: m a + (1 - m) b."""
        v = self.e(f"{m} * {a[0]} + (1.0 - {m}) * {b[0]}")
        g = self.total([f"{m} * {a[1]}" if a[1] else None, f"(1.0 - {m}) * {b[1]}" if b[1] else None])
        H = self.total([f"{m} * {a[2]}" if a[2] else None, f"(1.0 - {m}) * {b[2]}" if b[2] else None])
        return (v, g, H)

    def vec(self, comps):
        return Vec(self.p.emit(f"tack.vec3({', '.join(c[0] for c in comps)})", V)[0], comps)

    # --- grammar ----------------------------------------------------------

    def peek(self):
        return self.toks[self.k]

    def take(self):
        tok = self.toks[self.k]
        self.k += 1
        return tok

    def expect(self, val):
        _, v, pos = self.take()
        if v != val:
            self.p.error(f"expected '{val}'", pos)

    def parse(self):
        r = self.expr(0)
        kind, val, pos = self.peek()
        if kind != "end":
            self.p.error(f"unexpected '{val}'", pos)
        return r

    def expr(self, rbp):
        left = self.prefix(self.take())
        while True:
            kind, val, _ = self.peek()
            bp = P._INFIX[val][0] if kind == "op" and val in P._INFIX else 0
            if rbp >= bp:
                return left
            left = self.infix(self.take(), left)

    def is_vec(self, x):
        return isinstance(x, Vec)

    def scalar(self, x, pos):
        if self.is_vec(x):
            self.p.error("expected a scalar, got a vector", pos)
        return x

    def vector(self, x, pos):
        if not self.is_vec(x):
            self.p.error("expected a vector, got a scalar", pos)
        return x

    def prefix(self, tok):
        kind, val, pos = tok
        if kind == "num":
            return self.const(self.p.const(val)[0])
        if kind == "str":
            return self.field(val[1:-1], pos)
        if kind == "name":
            if self.peek()[1] == "(":
                return self.call(val, pos)
            return self.name(val, pos)
        if val == "(":
            r = self.expr(0)
            self.expect(")")
            return r
        if val == "{":
            parts = self.args("}")
            if len(parts) != 3:
                self.p.error("a vector needs three scalars", pos)
            return self.vec([self.scalar(x, pos) for x in parts])
        if val in ("-", "+", "!", "not"):
            a = self.expr(P._PREFIX_BP)
            if val == "+":
                return a
            if val == "-":
                neg = lambda s: (self.e(f"-{s[0]}"), self.e(f"-{s[1]}") if s[1] else None,
                                 self.e(f"-{s[2]}") if s[2] else None)
                return self.vec([neg(c) for c in a.comps]) if self.is_vec(a) else neg(a)
            return self.const(self.e(f"tack.select({self.scalar(a, pos)[0]} == 0.0, 1.0, 0.0)"))
        raise self.p.failure("the expression ends too early" if kind == "end" else f"unexpected '{val}'", pos)

    def name(self, val, pos):
        p = self.p
        if p.has_field(val):
            return self.field(val, pos)
        if val in ("pi", "true", "false"):
            return self.const(p.name(val, pos)[0])
        if val in ("iHat", "jHat", "kHat"):
            j = "ijk".index(val[0])
            return self.vec([self.const("1.0" if k == j else "0.0") for k in range(3)])
        if val in ("coords", "coordsX", "coordsY", "coordsZ"):
            x, _ = p.coords(pos)
            comps = [(self.e(f"{x}[{k}]"), "tack.vec3(" + ", ".join("1.0" if j == k else "0.0" for j in range(3)) + ")",
                      None) for k in range(3)]
            return self.vec(comps) if val == "coords" else comps["XYZ".index(val[-1])]
        base, _, comp = val.rpartition("_")
        if comp in ("X", "Y", "Z") and p.has_field(base):
            return self.vector(self.field(base, pos), pos).comps["XYZ".index(comp)]
        raise p.failure(f"unknown name '{val}'", pos)

    def leaf(self, name, field, getter, pos):
        """A scalar field with a basis: value, world gradient, world Hessian."""
        p = self.p
        pc = p.pc(pos)
        if any(g.count and g.shape.DIMENSION != 3 for g in p.data.topology.groups()):
            p.error("second derivatives are taken in solid cells; these are surfaces or lines", pos)
        p.uses_world_gradient = True
        u = p.arg("field", name, getter)
        if not ad.STENCIL:
            key = f"jet:{name}"
            if key not in p.loaded:
                k = len(p.loaded)
                p.loaded[key] = (f"a{k}", f"g{k}", f"h{k}")
                p.lines.append(f"a{k}, g{k}, h{k} = ad.world_jet(dom, {u}, c, {pc})")
            return p.loaded[key]
        v = p.load(f"value:{name}", f"{u}.value(c, {pc})", S)[0]
        g = p.load(f"grad:{name}", p.world_gradient(u, pc), V)[0]
        H = p.load(f"hess:{name}", f"ad.world_hessian(dom, {u}, c, {pc})", V)[0]
        return (v, g, H)

    def field(self, name, pos):
        p = self.p
        field = p.lookup(name, pos)
        kind = P._kind(field)
        if kind in ("hdiv", "hcurl"):
            p.error(f"second derivatives of {type(field.space).__name__} fields are not implemented", pos)
        value, t = p.field(name, pos)
        if t == P.M:
            p.error("second derivatives of tensor fields are not implemented", pos)
        if getattr(field, "gradient", None) is not None:
            p.error(f"'{name}' was given its gradient, not its second derivatives", pos)
        constant = kind == "values-cells" or isinstance(field.space, P.td.Constant)
        if kind != "basis" or constant:
            if t == S:
                return self.const(value)
            return self.vec([self.const(self.e(f"{value}[{k}]")) for k in range(3)])
        if t == S:
            return self.leaf(name, field, p.by_name(name), pos)
        whole = p.by_name(name)
        parts = p.once(f"components:{name}", lambda d, m: P._components(whole(d, m)))
        comps = [self.leaf(f"{name}#{k}", None, lambda d, m, k=k: parts(d, m)[k], pos) for k in range(3)]
        return Vec(value, comps)

    def args(self, close):
        parts = []
        if self.peek()[1] != close:
            parts.append(self.expr(0))
            while self.peek()[1] == ",":
                self.take()
                parts.append(self.expr(0))
        self.expect(close)
        return parts

    def call(self, fn, pos):
        p = self.p
        if fn in ("grad", "div", "curl", "qcrit", "laplacian"):
            p.error(f"{fn}() here would be a third derivative, which is not implemented", pos)
        if fn in ("cells", "fsum"):
            self.expect("(")
            kind, name, npos = self.take()
            name = name[1:-1] if kind == "str" else name
            self.expect(")")
            code, t = p.cells_of(name, npos) if fn == "cells" else p.fsum(name, npos)
            return self.const(code) if t == S else self.vec([self.const(self.e(f"{code}[{k}]")) for k in range(3)])
        if fn in ("points", "owner", "neighbor"):
            p.error(f"{fn}() has no derivative inside a cell", pos)
        self.expect("(")
        a = self.args(")")
        n = len(a)

        def want(count):
            if n != count:
                p.error(f"{fn}() takes {count} argument{'s' if count > 1 else ''}", pos)

        if fn in P._UNARY_S:
            want(1)
            x = self.scalar(a[0], pos)
            c = x[0]
            v = self.e(f"{P._UNARY_S[fn]}({c})")
            d = {
                "sqrt": (f"0.5 / {v}", f"-0.25 / ({v} * {v} * {v})"),
                "exp": (v, v),
                "ln": (f"1.0 / {c}", f"-1.0 / ({c} * {c})"),
                "log": (f"1.0 / {c}", f"-1.0 / ({c} * {c})"),
                "log10": (f"1.0 / ({c} * 2.302585092994046)", f"-1.0 / ({c} * {c} * 2.302585092994046)"),
                "sin": (f"cos({c})", f"-{v}"),
                "cos": (f"-sin({c})", f"-{v}"),
                "tan": (f"1.0 + {v} * {v}", f"2.0 * {v} * (1.0 + {v} * {v})"),
                "sinh": (f"cosh({c})", v),
                "cosh": (f"sinh({c})", v),
                "tanh": (f"1.0 - {v} * {v}", f"-2.0 * {v} * (1.0 - {v} * {v})"),
                "atan": (f"1.0 / (1.0 + {c} * {c})", f"-2.0 * {c} / ((1.0 + {c} * {c}) * (1.0 + {c} * {c}))"),
                "asin": (f"1.0 / sqrt(1.0 - {c} * {c})", f"{c} / ((1.0 - {c} * {c}) * sqrt(1.0 - {c} * {c}))"),
                "acos": (f"-1.0 / sqrt(1.0 - {c} * {c})", f"-{c} / ((1.0 - {c} * {c}) * sqrt(1.0 - {c} * {c}))"),
                "abs": (f"tm.sign({c})", None),
            }.get(fn)
            return self.chain(v, x, *d) if d else (v, None, None)
        if fn in ("min", "max"):
            want(2)
            x, y = self.scalar(a[0], pos), self.scalar(a[1], pos)
            m = self.e(f"tack.select({x[0]} {'<=' if fn == 'min' else '>='} {y[0]}, 1.0, 0.0)")
            return self.blend(m, x, y)
        if fn == "atan2":
            want(2)
            y, x = self.scalar(a[0], pos), self.scalar(a[1], pos)
            q = self.mul(y, self.recip(x))
            c = q[0]
            r = self.chain(self.e(f"atan2({y[0]}, {x[0]})"), q, f"1.0 / (1.0 + {c} * {c})",
                           f"-2.0 * {c} / ((1.0 + {c} * {c}) * (1.0 + {c} * {c}))")
            return r
        if fn == "pow":
            want(2)
            return self.power(self.scalar(a[0], pos), self.scalar(a[1], pos))
        if fn == "clamp":
            want(3)
            lo, x, hi = (self.scalar(t, pos) for t in a)
            m = self.e(f"tack.select({x[0]} > {lo[0]}, tack.select({x[0]} < {hi[0]}, 1.0, 0.0), 0.0)")
            v = self.e(f"min(max({x[0]}, {lo[0]}), {hi[0]})")
            g = self.e(f"{m} * {x[1]}") if x[1] else None
            H = self.e(f"{m} * {x[2]}") if x[2] else None
            return (v, g, H)
        if fn in ("dot", "mag2", "mag", "norm", "cross"):
            if fn == "dot":
                want(2)
                u, w = self.vector(a[0], pos), self.vector(a[1], pos)
                return self.sum([self.mul(u.comps[k], w.comps[k]) for k in range(3)])
            if fn == "cross":
                want(2)
                u, w = self.vector(a[0], pos), self.vector(a[1], pos)
                comps = [self.add(self.mul(u.comps[(k + 1) % 3], w.comps[(k + 2) % 3]),
                                  self.mul(u.comps[(k + 2) % 3], w.comps[(k + 1) % 3]), "-") for k in range(3)]
                return self.vec(comps)
            want(1)
            u = self.vector(a[0], pos)
            m2 = self.sum([self.mul(c, c) for c in u.comps])
            if fn == "mag2":
                return m2
            v = self.e(f"sqrt({m2[0]})")
            m = self.chain(v, m2, f"0.5 / {v}", f"-0.25 / ({v} * {v} * {v})")
            if fn == "mag":
                return m
            r = self.recip(m)
            return self.vec([self.mul(c, r) for c in u.comps])
        if fn == "if":
            want(3)
            return self.select(a[0], a[1], a[2], pos)
        if fn == "not":
            want(1)
            return self.const(self.e(f"tack.select({self.scalar(a[0], pos)[0]} == 0.0, 1.0, 0.0)"))
        raise p.failure(f"unknown function '{fn}'", pos)

    def sum(self, terms):
        r = terms[0]
        for t in terms[1:]:
            r = self.add(r, t)
        return r

    def power(self, a, b):
        if b[1] is None:
            e = b[0]
            v = self.e(f"{a[0]} ** {e}")
            return self.chain(v, a, f"{e} * {a[0]} ** ({e} - 1.0)", f"{e} * ({e} - 1.0) * {a[0]} ** ({e} - 2.0)")
        la = self.chain(self.e(f"log({a[0]})"), a, f"1.0 / {a[0]}", f"-1.0 / ({a[0]} * {a[0]})")
        x = self.mul(b, la)
        v = self.e(f"exp({x[0]})")
        return self.chain(v, x, v, v)

    def select(self, c, a, b, pos):
        c = self.scalar(c, pos)
        m = self.e(f"tack.select({c[0]} != 0.0, 1.0, 0.0)")
        if self.is_vec(a) != self.is_vec(b):
            self.p.error("both branches must have the same type", pos)
        if self.is_vec(a):
            return self.vec([self.blend(m, x, y) for x, y in zip(a.comps, b.comps)])
        return self.blend(m, a, b)

    def infix(self, tok, left):
        _, op, pos = tok
        bp, right_assoc = P._INFIX[op]
        if op == "?":
            a = self.expr(0)
            self.expect(":")
            b = self.expr(bp - 1)
            return self.select(left, a, b, pos)
        if op == "[":
            idx = self.expr(0)
            self.expect("]")
            k = self.p.const_value(self.scalar(idx, pos)[0])
            if k not in (0.0, 1.0, 2.0):
                self.p.error("a vector index must be 0, 1 or 2", pos)
            return self.vector(left, pos).comps[int(k)]
        right = self.expr(bp - 1 if right_assoc else bp)
        a, b = left, right
        if op in ("+", "-", "*", "/"):
            if self.is_vec(a) or self.is_vec(b):
                ca = a.comps if self.is_vec(a) else [a] * 3
                cb = b.comps if self.is_vec(b) else [b] * 3
                if op == "/" and self.is_vec(b) and not self.is_vec(a):
                    self.p.error("cannot divide a scalar by a vector", pos)
                return self.vec([self.binary(op, x, y) for x, y in zip(ca, cb)])
            return self.binary(op, a, b)
        a, b = self.scalar(a, pos), self.scalar(b, pos)
        if op == "^":
            return self.power(a, b)
        if op == "%":
            q = self.e(f"{a[0]} / {b[0]}")
            n = self.e(f"(floor({q}) if {q} >= 0.0 else ceil({q}))")
            return self.add(a, self.mul(b, self.const(n)), "-")
        if op in P._COMPARE:
            return self.const(self.e(f"tack.select({a[0]} {P._COMPARE[op]} {b[0]}, 1.0, 0.0)"))
        if op in ("and", "&&", "&"):
            return self.const(self.e(f"tack.select({a[0]} != 0.0, tack.select({b[0]} != 0.0, 1.0, 0.0), 0.0)"))
        return self.const(self.e(f"tack.select({a[0]} != 0.0, 1.0, tack.select({b[0]} != 0.0, 1.0, 0.0))"))

    def binary(self, op, a, b):
        if op in ("+", "-"):
            return self.add(a, b, op)
        if op == "*":
            return self.mul(a, b)
        return self.mul(a, self.recip(b))


def argument_tokens(toks, open_index, error, pos, fn):
    """The tokens of fn's parenthesized argument starting at open_index, with an end token,
    and the index after the closing parenthesis."""
    if toks[open_index][1] != "(":
        error(f"expected '(' after {fn}", pos)
    depth, i = 0, open_index
    while True:
        kind, val, vpos = toks[i]
        if kind == "end":
            error("missing ')'", vpos)
        depth += {"(": 1, ")": -1}.get(val, 0)
        if depth == 0:
            break
        i += 1
    return [*toks[open_index + 1:i], ("end", "", toks[i][2])], i + 1


def hessian_of(parent, toks, pos):
    """Compile a scalar expression's tokens with its second derivatives: (value, gradient,
    hessian), each code or None for zero."""
    r = Second(parent, toks).parse()
    if isinstance(r, Vec):
        parent.error("expected a scalar expression", pos)
    parent.pc(pos)
    return r


def vector_second(parent, toks, pos):
    """A vector expression's components with their second derivatives."""
    r = Second(parent, toks).parse()
    if not isinstance(r, Vec):
        parent.error("expected a vector expression", pos)
    parent.pc(pos)
    return r

