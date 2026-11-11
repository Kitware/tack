# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Forward-mode derivatives for grad(), div(), curl() and qcrit() of an expression.

The argument is compiled into the same kernel as its value, and every value carries
its derivative: a scalar its gradient (a 3-vector), a vector its Jacobian (a 3x3
matrix whose row i is the gradient of component i). Fields with a basis give theirs
through the basis and the geometry's Jacobian, as grad() of a field does; cell data
is constant in each cell and has none. The chain rule does the rest, so the
derivative is exact wherever the expression is differentiable, whatever the
fields' orders.

A derivative of None is zero, which keeps unneeded terms out of the kernel.
"""

from . import parser as P

S, V = P.S, P.V
ZERO_V = "tack.vec3(0.0, 0.0, 0.0)"
ZERO_M = "(0.0 * tack.identity(3))"


def _zero(t):
    return ZERO_V if t == S else ZERO_M


class Derivative:
    """Compiles one argument's tokens into the parent compiler's kernel."""

    def __init__(self, parent, toks):
        self.p = parent
        self.toks = toks
        self.k = 0
        parent.uses_derivatives = True

    # --- helpers --------------------------------------------------------

    def emit(self, code, t):
        return self.p.emit(code, t)[0]

    def d(self, code):
        return self.p.emit(code, S)[0]

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

    def error(self, message, pos):
        self.p.error(message, pos)

    def failure(self, message, pos):
        return self.p.failure(message, pos)

    def need(self, got, want, pos):
        self.p.need(got, want, pos)

    def sum(self, terms, t):
        terms = [term for term in terms if term is not None]
        return self.d(" + ".join(f"({term})" for term in terms)) if terms else None

    # --- grammar --------------------------------------------------------

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
            bp = P._INFIX[val][0] if kind == "op" and val in P._INFIX else 0
            if rbp >= bp:
                return left
            left = self.infix(self.take(), left)

    def prefix(self, tok):
        kind, val, pos = tok
        if kind == "num":
            code, _ = self.p.const(val)
            return code, S, None
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
            if len(parts) != 3 or any(t != S for _, t, _ in parts):
                self.error("a vector needs three scalars", pos)
            value = self.emit(f"tack.vec3({', '.join(c for c, _, _ in parts)})", V)
            rows = [d for _, _, d in parts]
            if all(r is None for r in rows):
                return value, V, None
            return value, V, self.d("tack.mat3(" + ", ".join(r or ZERO_V for r in rows) + ")")
        if val in ("-", "+", "!", "not"):
            a, t, da = self.expr(P._PREFIX_BP)
            if val == "+":
                return a, t, da
            if val == "-":
                return self.emit(f"-{a}", t), t, (self.d(f"-{da}") if da else None)
            self.need(t, S, pos)
            return self.emit(f"tack.select({a} == 0.0, 1.0, 0.0)", S), S, None
        raise self.failure("the expression ends too early" if kind == "end" else f"unexpected '{val}'", pos)

    def name(self, val, pos):
        p = self.p
        if p.has_field(val):
            return self.field(val, pos)
        if val in ("iHat", "jHat", "kHat", "pi", "true", "false"):
            code, t = p.name(val, pos)
            return code, t, None
        if val == "coords":
            x, _ = p.coords(pos)
            return x, V, self.d("tack.identity(3)")
        if val in ("coordsX", "coordsY", "coordsZ"):
            x, _ = p.coords(pos)
            k = "XYZ".index(val[-1])
            unit = ", ".join("1.0" if j == k else "0.0" for j in range(3))
            return self.emit(f"{x}[{k}]", S), S, f"tack.vec3({unit})"
        base, _, comp = val.rpartition("_")
        if comp in ("X", "Y", "Z") and p.has_field(base):
            a, t, J = self.field(base, pos)
            self.need(t, V, pos)
            k = "XYZ".index(comp)
            return self.emit(f"{a}[{k}]", S), S, (self.d(f"ad.row({J}, {k})") if J else None)
        raise self.failure(f"unknown name '{val}'", pos)

    def field(self, name, pos):
        p = self.p
        field = p.lookup(name, pos)
        value, t = p.field(name, pos)
        kind = P._kind(field)
        if t == P.M:
            # A tensor's derivative is three matrices, one per direction: slice m holds d/dx_m of
            # each entry. Only indexing and div() take it; the front end writes the rest out by entry.
            if kind != "basis" or isinstance(field.space, P.td.Constant):
                return value, t, None
            g = p.tensor_gradients(name, pos)
            return value, t, tuple(self.d("tack.mat3(" + ", ".join(f"{g[k]}[{m}]" for k in range(9)) + ")")
                                   for m in range(3))
        if kind in ("hdiv", "hcurl"):
            self.error(f"derivatives of {type(field.space).__name__} fields inside an expression "
                       "are not implemented", pos)
        constant = kind == "values-cells" or isinstance(field.space, P.td.Constant)
        given = getattr(field, "gradient", None) is not None
        if kind != "basis" or (constant and not given):
            return value, t, None
        pc = p.pc(pos)
        p.uses_world_gradient = True
        if t == S:
            u = p.arg("field", name, p.by_name(name))
            return value, S, p.load(f"grad:{name}", p.world_gradient(u, pc), V)[0]
        whole = p.by_name(name)
        parts = p.once(f"components:{name}", lambda d, m: P._components(whole(d, m)))
        rows = []
        for k in range(3):
            u = p.arg("field", f"{name}#{k}", lambda d, m, k=k: parts(d, m)[k])
            rows.append(p.load(f"grad:{name}#{k}", p.world_gradient(u, pc), V)[0])
        return value, V, p.load(f"jac:{name}", f"tack.mat3({', '.join(rows)})", V)[0]

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

    def call(self, fn, pos):
        p = self.p
        if fn in ("grad", "div", "curl"):
            return self.second(fn, pos)
        if fn == "laplacian":
            self.error("laplacian() inside a derivative is a third derivative, which is not implemented", pos)
        if fn == "qcrit":
            self.error("qcrit() inside a derivative is not implemented; take derivatives of its terms", pos)
        if fn in ("cells", "fsum"):
            name, npos = self.field_arg(fn)
            code, t = p.cells_of(name, npos) if fn == "cells" else p.fsum(name, npos)
            return code, t, None            # constant in each cell
        if fn in ("points", "owner", "neighbor"):
            self.error(f"{fn}() has no derivative inside a cell", pos)
        self.expect("(")
        a = self.args_list(")")
        c = [x for x, _, _ in a]
        t = [x for _, x, _ in a]
        g = [x for _, _, x in a]
        if any(isinstance(x, tuple) for x in g):
            self.error(f"{fn}() of a tensor inside a derivative is not implemented; index the tensor", pos)

        def check(*want):
            if len(a) != len(want):
                self.error(f"{fn}() takes {len(want)} argument{'s' if len(want) > 1 else ''}", pos)
            for got, w in zip(t, want):
                self.need(got, w, pos)

        if fn in P._UNARY_S:
            check(S)
            x = c[0]
            v = self.emit(f"{P._UNARY_S[fn]}({x})", S)
            factor = {
                "abs": f"tm.sign({x})", "acos": f"-1.0 / sqrt(1.0 - {x} * {x})",
                "asin": f"1.0 / sqrt(1.0 - {x} * {x})", "atan": f"1.0 / (1.0 + {x} * {x})",
                "cos": f"-sin({x})", "cosh": f"sinh({x})", "exp": v, "ln": f"1.0 / {x}",
                "log": f"1.0 / {x}", "log10": f"1.0 / ({x} * 2.302585092994046)", "sin": f"cos({x})",
                "sinh": f"cosh({x})", "sqrt": f"0.5 / {v}", "tan": f"1.0 / (cos({x}) * cos({x}))",
                "tanh": f"1.0 - {v} * {v}",
            }.get(fn)
            if factor is None or g[0] is None:
                return v, S, None
            return v, S, self.d(f"({factor}) * {g[0]}")
        if fn in ("min", "max"):
            check(S, S)
            v = self.emit(f"{fn}({c[0]}, {c[1]})", S)
            if g[0] is None and g[1] is None:
                return v, S, None
            m = self.emit(f"tack.select({c[0]} {'<=' if fn == 'min' else '>='} {c[1]}, 1.0, 0.0)", S)
            return v, S, self.sum([f"{m} * {g[0]}" if g[0] else None,
                                   f"(1.0 - {m}) * {g[1]}" if g[1] else None], S)
        if fn == "atan2":
            check(S, S)
            y, x = c
            v = self.emit(f"atan2({y}, {x})", S)
            r = self.emit(f"{x} * {x} + {y} * {y}", S)
            return v, S, self.sum([f"{x} * {g[0]} / {r}" if g[0] else None,
                                   f"-{y} * {g[1]} / {r}" if g[1] else None], S)
        if fn == "pow":
            check(S, S)
            return self.power(c[0], g[0], c[1], g[1])
        if fn in ("mag", "mag2", "norm"):
            check(V)
            x, J = c[0], g[0]
            if fn == "mag":
                v = self.emit(f"{x}.norm()", S)
                return v, S, (self.d(f"({J}.transpose() @ {x}) / {v}") if J else None)
            if fn == "mag2":
                v = self.emit(f"{x}.norm_sqr()", S)
                return v, S, (self.d(f"2.0 * ({J}.transpose() @ {x})") if J else None)
            v = self.emit(f"{x}.normalized()", V)
            return v, V, (self.d(f"ad.unit_jac({x}, {J})") if J else None)
        if fn == "dot":
            check(V, V)
            v = self.emit(f"{c[0]}.dot({c[1]})", S)
            return v, S, self.sum([f"{g[0]}.transpose() @ {c[1]}" if g[0] else None,
                                   f"{g[1]}.transpose() @ {c[0]}" if g[1] else None], S)
        if fn == "cross":
            check(V, V)
            v = self.emit(f"{c[0]}.cross({c[1]})", V)
            if g[0] is None and g[1] is None:
                return v, V, None
            return v, V, self.d(f"ad.cross_jac({c[0]}, {g[0] or ZERO_M}, {c[1]}, {g[1] or ZERO_M})")
        if fn == "if":
            if len(a) != 3:
                self.error("if() takes 3 arguments", pos)
            return self.select(a[0], a[1], a[2], pos)
        if fn == "clamp":
            check(S, S, S)
            v = self.emit(f"min(max({c[1]}, {c[0]}), {c[2]})", S)
            if all(x is None for x in g):
                return v, S, None
            # The derivative of whichever of lo, x and hi the value is.
            top = self.emit(f"tack.select(max({c[1]}, {c[0]}) >= {c[2]}, 1.0, 0.0)", S)
            above = self.emit(f"(1.0 - {top}) * tack.select({c[1]} > {c[0]}, 1.0, 0.0)", S)
            return v, S, self.sum([f"(1.0 - {top} - {above}) * {g[0]}" if g[0] else None,
                                   f"{above} * {g[1]}" if g[1] else None,
                                   f"{top} * {g[2]}" if g[2] else None], S)
        if fn == "not":
            check(S)
            return self.emit(f"tack.select({c[0]} == 0.0, 1.0, 0.0)", S), S, None
        if fn in ("trace", "det", "transpose", "inv", "sym", "skew", "dev", "outer", "ddot", "vonmises", "eigvals"):
            raise self.failure(f"derivatives of tensor expressions such as {fn}() are not implemented", pos)
        raise self.failure(f"unknown function '{fn}'", pos)

    def second(self, fn, pos):
        """grad, div or curl of an expression inside a derivative: its value from the
        expression's gradients, its derivative from their Hessians."""
        from ._second import argument_tokens, hessian_of, vector_second
        inner, self.k = argument_tokens(self.toks, self.k, self.error, pos, fn)
        if fn == "grad":
            _, g, H = hessian_of(self.p, inner, pos)
            return (g or ZERO_V), V, H
        comps = vector_second(self.p, inner, pos).comps
        g = [c[1] for c in comps]
        H = [c[2] for c in comps]

        def row(k, i):
            return f"ad.row({H[k]}, {i})" if H[k] else None

        if fn == "div":
            terms = [f"{g[i]}[{i}]" for i in range(3) if g[i]]
            value = self.emit(" + ".join(terms), S) if terms else "0.0"
            rows = [row(i, i) for i in range(3)]
            return value, S, self.sum(rows, S)
        # curl: component k is d/dx_(k+1) of V_(k+2) - d/dx_(k+2) of V_(k+1).
        parts, rows = [], []
        for k in range(3):
            a, b = (k + 2) % 3, (k + 1) % 3
            terms = [f"{g[a]}[{b}]" if g[a] else None, f"-{g[b]}[{a}]" if g[b] else None]
            terms = [t for t in terms if t]
            parts.append(" + ".join(terms) if terms else "0.0")
            r = [row(a, b), f"-({row(b, a)})" if row(b, a) else None]
            rows.append(self.sum(r, S) or ZERO_V)
        value = self.emit(f"tack.vec3({', '.join(parts)})", V)
        if all(r == ZERO_V for r in rows):
            return value, V, None
        return value, V, self.d(f"tack.mat3({', '.join(rows)})")

    def power(self, a, da, b, db):
        v = self.emit(f"{a} ** {b}", S)
        return v, S, self.sum([f"{b} * {a} ** ({b} - 1.0) * {da}" if da else None,
                               f"{v} * log({a}) * {db}" if db else None], S)

    def infix(self, tok, left):
        _, op, pos = tok
        bp, right_assoc = P._INFIX[op]
        if op == "?":
            a = self.expr(0)
            self.expect(":")
            b = self.expr(bp - 1)
            return self.select(left, a, b, pos)
        if op == "[":
            idx, _, _ = self.expr(0)
            self.expect("]")
            k = self.p.const_value(idx)
            if k not in (0.0, 1.0, 2.0):
                self.error("an index must be 0, 1 or 2", pos)
            if left[1] == P.M:
                a, D, k = left[0], left[2], int(k)
                row = self.emit(f"tack.vec3({a}[{k}, 0], {a}[{k}, 1], {a}[{k}, 2])", V)
                if not D:
                    return row, V, None
                rows = ", ".join(f"tack.vec3({Dm}[{k}, 0], {Dm}[{k}, 1], {Dm}[{k}, 2])" for Dm in D)
                return row, V, self.d(f"tack.mat3({rows}).transpose()")
            self.need(left[1], V, pos)
            J = left[2]
            return self.emit(f"{left[0]}[{int(k)}]", S), S, (self.d(f"ad.row({J}, {int(k)})") if J else None)
        (a, ta, da) = left
        (b, tb, db) = self.expr(bp - 1 if right_assoc else bp)
        if isinstance(da, tuple) or isinstance(db, tuple):
            self.error("a tensor inside a derivative is read entry by entry: index it, or take div() of it", pos)
        if op in ("+", "-"):
            t = V if V in (ta, tb) else S
            v = self.emit(f"{a} {op} {b}", t)
            # A scalar beside a vector adds its gradient to every row.
            da = f"ad.rows3({da})" if (da and t == V and ta == S) else da
            db = f"ad.rows3({db})" if (db and t == V and tb == S) else db
            if op == "-" and db:
                db = f"-({db})"
            return v, t, self.sum([da, db], t)
        if op == "*":
            t = V if V in (ta, tb) else S
            v = self.emit(f"{a} * {b}", t)
            if ta == tb == S:
                return v, S, self.sum([f"{a} * {db}" if db else None, f"{b} * {da}" if da else None], S)
            if ta == tb == V:
                return v, V, self.sum([f"ad.diag_mul({a}, {db})" if db else None,
                                       f"ad.diag_mul({b}, {da})" if da else None], V)
            (s, ds), (w, dw) = ((a, da), (b, db)) if ta == S else ((b, db), (a, da))
            return v, V, self.sum([f"{s} * {dw}" if dw else None,
                                   f"{w}.outer_product({ds})" if ds else None], V)
        if op == "/":
            if tb == V and ta == S:
                self.error("cannot divide a scalar by a vector", pos)
            t = ta
            v = self.emit(f"{a} / {b}", t)
            if ta == tb == V:
                if da is None and db is None:
                    return v, V, None
                return v, V, self.d(f"ad.div_vv({a}, {da or ZERO_M}, {b}, {db or ZERO_M})")
            if ta == S:
                return v, S, self.sum([f"{da} / {b}" if da else None,
                                       f"-{a} * {db} / ({b} * {b})" if db else None], S)
            return v, V, self.sum([f"{da} / {b}" if da else None,
                                   f"-{a}.outer_product({db}) / ({b} * {b})" if db else None], V)
        self.need(ta, S, pos)
        self.need(tb, S, pos)
        if op == "^":
            return self.power(a, da, b, db)
        if op == "%":
            q = self.emit(f"{a} / {b}", S)
            n = self.emit(f"(floor({q}) if {q} >= 0.0 else ceil({q}))", S)
            v = self.emit(f"{a} - {b} * {n}", S)
            return v, S, self.sum([da, f"-{n} * {db}" if db else None], S)
        if op in P._COMPARE:
            return self.emit(f"tack.select({a} {P._COMPARE[op]} {b}, 1.0, 0.0)", S), S, None
        if op in ("and", "&&", "&"):
            return self.emit(f"tack.select({a} != 0.0, tack.select({b} != 0.0, 1.0, 0.0), 0.0)", S), S, None
        return self.emit(f"tack.select({a} != 0.0, 1.0, tack.select({b} != 0.0, 1.0, 0.0))", S), S, None

    def select(self, c, a, b, pos):
        self.need(c[1], S, pos)
        if a[1] != b[1]:
            self.error("both branches must have the same type", pos)
        t = a[1]
        v = self.emit(f"tack.select({c[0]} != 0.0, {a[0]}, {b[0]})", t)
        if a[2] is None and b[2] is None:
            return v, t, None
        m = self.emit(f"tack.select({c[0]} != 0.0, 1.0, 0.0)", S)
        return v, t, self.sum([f"{m} * {a[2]}" if a[2] else None,
                               f"(1.0 - {m}) * {b[2]}" if b[2] else None], t)


def compile_argument(parent, fn, pos):
    """Compile fn's argument, an expression, with its derivative; returns (value, type, derivative)."""
    toks = parent.toks
    open_index = parent.k
    if toks[open_index][1] != "(":
        parent.error(f"expected '(' after {fn}", pos)
    depth, i = 0, open_index
    while True:
        kind, val, vpos = toks[i]
        if kind == "end":
            parent.error("missing ')'", vpos)
        depth += {"(": 1, ")": -1}.get(val, 0)
        if depth == 0:
            break
        i += 1
    inner = [*toks[open_index + 1:i], ("end", "", toks[i][2])]
    parent.k = i + 1
    if parent.mode not in ("cells", "nodes"):
        parent.error(f"{fn}() is for cell and node expressions; a point gradient needs "
                     "averaging over cells (td.gradients(at='points'))", pos)
    parent.pc(pos)
    return Derivative(parent, inner).parse()

