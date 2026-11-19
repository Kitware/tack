# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Generate tack/dataset/_clip_snap_tables.py from the clip case tables.

A clip through a mesh point -- a corner at the value, within round-off -- cuts
the edges from it to the corners on the dropped side at that corner. The case
tables (``_clip_tables``) name each such crossing by its edge, so it lands on top
of the corner as a point of its own, and the pieces around it have no volume or
name a place twice. These tables give, for each case and each set of a cut cell's
corners at the value (always above it, whichever side is kept: ``clip``'s
``_cell_case``), the case's pieces as they are once each such crossing is the
corner itself:

- a piece left with no volume (no area, for a polygon; no length, for a line) is
  dropped;
- a piece some of whose points have come together becomes the shape it is now:
  a tetrahedron, pyramid or wedge (or a triangle), wound as VTK's shapes are;
- a hexahedron that is no shape -- one edge collapsed (four quadrilaterals and
  two triangles), or, from fields no linear field gives, two opposite edges or
  three side by side -- stays a hexahedron, its collapsed corners one point
  named twice, as VTK and finite-element codes write a collapsed hexahedron;
- a centroid no piece uses any more is dropped with them.

Whether a piece has volume is decided on the reference cell with corner values at
random, so a piece is dropped only where it is flat for every field; the
generator checks that none is flat only where the field is linear across the
cell (a plane's distance), on a linear field giving the case and set where one
does, at a point inside the fields that do (an average of linear programs'
corners), so a plane's clip has no piece of no volume either. Corner
order, edges and entries are the case tables'; a wedge is in VisIt's order, as
the case tables' are, so ``clip`` reads both alike.

Usage (from the repository root):
  python packages/tack-vis/tools/make_clip_snap_tables.py \
      packages/tack-vis/src/tack/dataset/_clip_snap_tables.py

Needs NumPy and SciPy; the result is the same on every run (fixed seeds).
"""
import itertools
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "packages" / "tack-core" / "src"))
from tack.dataset import _clip_tables as T

NPTS = {1: 1, 3: 2, 5: 3, 9: 4, 10: 4, 12: 8, 13: 6, 14: 5}
DIM = {1: 0, 3: 1, 5: 2, 9: 2, 10: 3, 12: 3, 13: 3, 14: 3}
# Faces wound out of each shape, in VTK's corner order.
FACES = {10: [(0, 1, 3), (1, 2, 3), (2, 0, 3), (0, 2, 1)],
         14: [(0, 3, 2, 1), (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)],
         13: [(0, 1, 2), (3, 5, 4), (0, 3, 4, 1), (1, 4, 5, 2), (2, 5, 3, 0)],
         12: [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6),
              (3, 0, 4, 7)]}
# The reference cells, by table, in the tables' corner order (VisIt's wedge
# is VTK's corners 0 2 1 3 5 4).
REFERENCE = {
    1: [(0, 0, 0)],
    3: [(0, 0, 0), (1, 0, 0)],
    5: [(0, 0, 0), (1, 0, 0), (0, 1, 0)],
    9: [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)],
    10: [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)],
    12: [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1),
         (0, 1, 1)],
    13: [(0, 0, 0), (0, 1, 0), (1, 0, 0), (0, 0, 1), (0, 1, 1), (1, 0, 1)],
    14: [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0.5, 0.5, 1)],
}
WEDGE = (0, 2, 1, 3, 5, 4)          # VisIt's wedge to VTK's, and back
CENTROID = T.CENTROID


def records(table, invert, case):
    data = T.DATA[invert]
    at = T.CASES[invert][table][case]
    out = []
    for _ in range(data[at]):
        kind, count = data[at + 1], data[at + 2]
        out.append((kind, list(data[at + 3:at + 3 + count])))
        at += 2 + count
    return out


def _volume(points):
    a, b, c, d = points
    return float(np.linalg.det(np.array([b - a, c - a, d - a])))


def _signed(kind, ids, pos):
    """A shape's signed volume by its tetrahedra, VTK's corner order."""
    p = [pos[i] for i in ids]
    if kind == 10:
        return _volume(p)
    if kind == 14:
        return _volume([p[0], p[1], p[2], p[4]]) + _volume([p[0], p[2], p[3], p[4]])
    if kind == 13:
        return (_volume([p[0], p[1], p[2], p[3]]) + _volume([p[1], p[2], p[3], p[4]])
                + _volume([p[2], p[3], p[4], p[5]]))
    if kind == 12:
        tets = [(0, 1, 3, 4), (1, 2, 3, 6), (1, 4, 5, 6), (3, 6, 7, 4), (1, 3, 4, 6)]
        return sum(_volume([p[i] for i in t]) for t in tets)
    raise ValueError(kind)


def _rank(points):
    if len(points) < 2:
        return 0
    m = np.array(points[1:]) - points[0]
    s = np.linalg.svd(m, compute_uv=False)
    return int(np.sum(s > 1e-9 * max(1.0, s[0])))


def reduced(kind, ids, pos):
    """``(kind, ids)`` of a piece once its points have come together (VTK's order),
    or ``None`` for a piece with nothing left."""
    distinct = list(dict.fromkeys(ids))
    if _rank([pos[i] for i in distinct]) < DIM[kind]:
        return None
    if len(distinct) == len(ids):
        return kind, ids
    if DIM[kind] == 2:
        assert kind == 9 and len(distinct) == 3, (kind, ids)
        return 5, [ids[i] for i in range(4) if ids[i] != ids[i - 1]]
    assert DIM[kind] == 3, (kind, ids)
    faces = []
    for f in FACES[kind]:
        v = [ids[k] for k in f]
        w = [v[i] for i in range(len(v)) if v[i] != v[i - 1]]
        if len(set(w)) >= 3:
            assert len(set(w)) == len(w), (kind, ids, w)
            faces.append(w)
    sizes = sorted(len(f) for f in faces)
    if len(distinct) == 4:
        tet = distinct
        if _volume([pos[i] for i in tet]) < 0:
            tet = [tet[0], tet[2], tet[1], tet[3]]
        return 10, tet
    if len(distinct) == 5 and sizes == [3, 3, 3, 3, 4]:
        base = next(f for f in faces if len(f) == 4)
        apex = next(i for i in distinct if i not in base)
        out = [*base, apex]
        if _signed(14, out, pos) < 0:
            out = [base[0], base[3], base[2], base[1], apex]
        return 14, out
    if len(distinct) == 6 and sizes == [3, 3, 4, 4, 4]:
        t1, t2 = (f for f in faces if len(f) == 3)
        edges = {frozenset((f[i], f[i - 1])) for f in faces if len(f) == 4
                 for i in range(4)}
        partner = [next(b for b in t2 if frozenset((a, b)) in edges) for a in t1]
        out = [*t1, *partner]
        if _signed(13, out, pos) < 0:
            out = [t1[0], t1[2], t1[1], partner[0], partner[2], partner[1]]
        return 13, out
    if kind == 12:
        return 12, ids                 # a collapsed hexahedron: no shape is it
    raise AssertionError(f"no shape for {kind} {ids} faces {faces}")


def realization(corners, above, at, rng):
    """Corner values of a linear field giving the case and the set at the value, at
    a point inside the fields that do; ``None`` where no linear field does."""
    n = len(corners)
    rows = [[*p, 1.0] for p in corners]
    eq = [rows[k] for k in at]
    ub, bound = [], []
    for k in range(n):
        if k in at:
            continue
        ub.append([-v for v in rows[k]] if above[k] else rows[k])
        bound.append(-1.0)
    found = []
    for _ in range(4):
        r = linprog(rng.normal(size=4), A_ub=ub or None, b_ub=bound or None,
                    A_eq=eq or None, b_eq=[0.0] * len(eq) or None, bounds=[(-20, 20)] * 4,
                    method="highs")
        if not r.success:
            return None
        found.append(r.x)
    a = np.mean(found, axis=0)
    values = np.array(corners) @ a[:3] + a[3]
    values[list(at)] = 0.0
    assert all((values[k] > 0) == bool(above[k]) for k in range(n) if k not in at)
    return values


def _positions(table, at, values, snap_edges=True):
    """Corner and crossing positions on the reference cell for corner ``values``."""
    edges = T.EDGES[table]
    corners = np.array(REFERENCE[table], float)
    pos = {k: corners[k] for k in range(len(corners))}
    for e in range(len(edges) // 2):
        a, b = edges[2 * e], edges[2 * e + 1]
        if values[a] != values[b] and (values[a] > 0) != (values[b] > 0):
            t = -values[a] / (values[b] - values[a])
            pos[8 + e] = corners[a] + t * (corners[b] - corners[a])
    return pos


def snapped_records(table, invert, case, at, generic, linear=None):
    """The case's records with each crossing at a corner of ``at`` made that
    corner, its pieces reduced (``reduced``) on ``generic`` corner values (values
    no linear field need give: a piece flat there is flat for every field), unused
    centroids dropped; no piece may be flat only on ``linear`` values (a linear
    field's), whose clip would keep it with no volume."""
    edges = T.EDGES[table]

    def snap(entry):
        if entry < T.FIRST_EDGE or entry == CENTROID:
            return entry
        a, b = edges[2 * (entry - 8)], edges[2 * (entry - 8) + 1]
        if a in at:
            return a
        if b in at:
            return b
        return entry

    given = records(table, invert, case)
    pos = {"g": _positions(table, at, generic)}
    if linear is not None:
        pos["l"] = _positions(table, at, linear)
    for kind, entries in given:
        if kind == 0:
            for p in pos.values():
                p[CENTROID] = np.mean([p[i] for i in map(snap, entries)], axis=0)
    out = []
    for kind, entries in given:
        if kind == 0:
            out.append((0, [snap(e) for e in entries]))
            continue
        vtk = [entries[i] for i in WEDGE] if kind == 13 else entries
        piece = reduced(kind, [snap(e) for e in vtk], pos["g"])
        if piece is None:
            continue
        kind, ids = piece
        assert "l" not in pos or _rank([pos["l"][i] for i in dict.fromkeys(ids)]) == DIM[kind], (
            "flat only where the field is linear", table, invert, case, at, kind, ids)
        out.append((kind, [ids[i] for i in WEDGE] if kind == 13 else ids))
    if not any(CENTROID in ids for kind, ids in out if kind):
        out = [(kind, ids) for kind, ids in out if kind]
    return given, out


def check_winding(table, rng):
    """Every 3D piece of every case, with no corner at the value, has positive
    volume in VTK's order: the reference cells and the wedge order are right."""
    n = NPTS[table]
    for invert in (False, True):
        for case in range(1, (1 << n) - 1):
            above = [(case >> k) & 1 for k in range(n)]
            values = np.where(np.array(above) == 1, rng.random(n) + 0.2, -rng.random(n) - 0.2)
            given, _ = snapped_records(table, invert, case, (), values)
            corners = np.array(REFERENCE[table], float)
            pos = {k: corners[k] for k in range(n)}
            edges = T.EDGES[table]
            for e in range(len(edges) // 2):
                a, b = edges[2 * e], edges[2 * e + 1]
                t = -values[a] / (values[b] - values[a])
                pos[8 + e] = corners[a] + t * (corners[b] - corners[a])
            for kind, entries in given:
                if kind == 0:
                    pos[CENTROID] = np.mean([pos[i] for i in entries], axis=0)
            for kind, entries in given:
                if kind and DIM[kind] == 3:
                    vtk = [entries[i] for i in WEDGE] if kind == 13 else entries
                    assert _signed(kind, vtk, pos) > 0, (table, invert, case, kind, entries)


def flat(pieces):
    out = [len(pieces)]
    for kind, ids in pieces:
        out += [kind, len(ids), *ids]
    return out


def _blob(values, dtype):
    import base64
    import zlib

    raw = np.asarray(values, dtype).tobytes()
    return base64.b64encode(zlib.compress(raw, 9)).decode("ascii")


def write(target, data, cases):
    """The module: the records as u8 (every entry is under 256) and each shape's
    pairs as i32, zlib-compressed and base64-encoded, 130 KB where the lists were
    1.1 MB, decoded on first use."""
    assert all(max(d) < 256 for d in data.values())
    lines = [
        "# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.",
        "# SPDX-License-Identifier: BSD-3-Clause",
        "",
        '"""Clip case records for cut cells with corners at the value: generated by',
        "packages/tack-vis/tools/make_clip_snap_tables.py from ``_clip_tables``; do not",
        "edit (the generator's docstring says how they are made).",
        "",
        "A crossing on an edge from a corner at the value is that corner; pieces left",
        "with no volume are dropped and the others are the shape they become (a",
        "hexahedron no shape fits stays one, a corner named twice). ``data(invert)``",
        "holds records as ``_clip_tables.DATA`` does (u8);",
        "``cases(invert, shape)`` is pairs ``(case << n | at, offset)`` (i32), ``n`` the",
        "shape's corners and bit ``k`` of ``at`` set where corner ``k`` is at the value,",
        "for each case and set whose records differ from the case tables'.",
        '"""',
        "",
        "import base64",
        "import functools",
        "import zlib",
        "",
        "import numpy as np",
        "",
        "_DATA = {",
        *(f"    {inv}: \"{_blob(data[inv], np.uint8)}\"," for inv in (False, True)),
        "}",
        "",
        "_CASES = {",
    ]
    for inv in (False, True):
        lines.append(f"    {inv}: {{")
        for table in sorted(cases[inv]):
            if cases[inv][table]:
                lines.append(f"        {table}: \"{_blob(cases[inv][table], np.int32)}\",")
        lines.append("    },")
    lines += [
        "}",
        "",
        "",
        "def _decoded(blob, dtype):",
        "    return np.frombuffer(zlib.decompress(base64.b64decode(blob)), dtype)",
        "",
        "",
        "@functools.cache",
        "def data(invert):",
        "    return _decoded(_DATA[invert], np.uint8)",
        "",
        "",
        "@functools.cache",
        "def cases(invert, shape):",
        "    blob = _CASES[invert].get(shape)",
        "    return np.zeros((0,), np.int32) if blob is None else _decoded(blob, np.int32)",
        "",
    ]
    Path(target).write_text("\n".join(lines))


def main(target):
    rng = np.random.default_rng(309)
    data = {False: [], True: []}
    cases = {False: {}, True: {}}
    stats = {}
    for table in sorted(T.EDGES):
        n = NPTS[table]
        corners = np.array(REFERENCE[table], float)
        if DIM[table] == 3:
            check_winding(table, rng)
        for invert in (False, True):
            index = []
            seen = {}
            for case in range(1 << n):
                above = [(case >> k) & 1 for k in range(n)]
                # Corners at the value are above it whichever side is kept (clip's
                # _cell_case), so the two sides take complementary cases. With every
                # corner above at the value, the cell below is kept whole, as its
                # pieces (clip drops it outright on the side above: it only touches).
                ones = [k for k in range(n) if above[k]]
                if not ones or len(ones) == n:
                    continue
                for r in range(1, len(ones) + 1):
                    for at in itertools.combinations(ones, r):
                        linear = realization(corners, above, at, rng)
                        generic = np.where(np.array(above) == 1, rng.random(n) + 0.2,
                                           -rng.random(n) - 0.2)
                        generic[list(at)] = 0.0
                        given, out = snapped_records(table, invert, case, at, generic, linear)
                        if out == given:
                            continue
                        record = tuple(flat(out))
                        if record not in seen:
                            seen[record] = len(data[invert])
                            data[invert].extend(record)
                        mask = sum(1 << k for k in at)
                        index += [(case << n) | mask, seen[record]]
                        key = (table, "linear" if linear is not None else "not linear")
                        stats[key] = stats.get(key, 0) + 1
            cases[invert][table] = index

    write(target, data, cases)
    for key in sorted(stats):
        print(key, stats[key])
    print("data:", len(data[False]), len(data[True]))


if __name__ == "__main__":
    main(sys.argv[1])
