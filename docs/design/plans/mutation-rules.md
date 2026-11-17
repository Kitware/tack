# Mutation rules, stage 1

**Status: proposed.** Issue [#323](https://github.com/Kitware/tack/issues/323),
2026-11-10. Berk Geveci has said yes to every recommendation (D1–D6); this
is for review by the others. Revised 2026-11-16 after Jeff Lee's review, for
in-situ use: a caller's own structure is no longer sealed (D2), writes Tack
cannot see have an explicit call (D4), a sender may keep its buffer (D5), and
the seal's limit at DLPack is stated (D1).

## Summary

The structure Tack builds (a topology's derived faces, edges and links, its
spaces' tables) becomes read-only once it is built, for every array over that
memory, not just one. Structure a caller passes in stays the caller's to
write, through its own arrays, and the topology notices when it changes.
Values (positions, field values, ghost marks) stay writable, and every cache
derived from them notices when they change. An array sent through
`tack.communication` becomes read-only to its sender unless it asks to keep
it. Nothing is copied: sharing memory with other libraries stays as it is.

This is the first stage of a longer direction: a completed dataset immutable
until it is explicitly reset. Stage 1 does not make values immutable, does not
add dataset generations or a reset, and does not freeze field mappings. Those
wait for the prerequisites listed under [Later stages](#later-stages).

## Why

Tack derives and keeps a lot from a dataset: faces, edges, point links, cell
groups, cell points, boundary surfaces, cell locators, ghost exchange plans.
An inventory of `tack.dataset` (at `4756ae3d`) found about forty such caches.
Five check whether what they were derived from has changed, all of them by
the values' revision (`arrays._revision`): the cell and point locators
(`serves()`), the measured cell geometry (`_mesh_kept`), the boundary
surface's geometry, and the serendipity spaces' full values. None of the
caches derived from structure checks anything. So this runs and gives wrong
answers silently:

```python
topology = td.UnstructuredTopology(types, offsets, connectivity)
faces = topology.faces()          # cached on the topology
connectivity.fill(0)              # or a kernel writing it
td.extract_surface(data)          # uses the faces of the old connectivity
```

The caller's `connectivity` is the topology's own array (constructors keep a
Tack array as given), and several derived arrays share its memory: a
single-shape group's rows are `connectivity.reshape(...)`, `point_links`'
keys are `connectivity`, an `L2` space's offsets are `topology.offsets`, a
polygon topology's faces' `side_face` is its `cell_faces`. A write to one
reaches all of them.

## What exists today

- **Read-only arrays.** `Array._writable` is false for an `array_from_ptr`
  import (by default) and a read-only DLPack import. Host writes
  (`Array._check_writable`: `fill`, `copyto`, uploads) and kernel stores
  (dispatch's `check_writable_arrays`, on every call, over only the arrays
  the kernel may store to) refuse them. The flag is on one `Array` object,
  so another array over the same memory (a `reshape`, a scalar view) can
  still write.
- **Write counts.** `DeviceBuffer.version` counts every write Tack makes to a
  buffer, shared by every array over it, and `arrays._revision` turns it into
  a token. Writes Tack does not make (through memory shared by DLPack or
  `wrap_ptr`) are not counted.
- **Zero-copy imports are read-only already.** The VTK interop's `copy=False`
  imports offsets, connectivity, types and points as read-only arrays.

## Decisions

### D1. Seal structure at the buffer

`DeviceBuffer` gets a one-way `sealed` flag: set once, never cleared, shared
by every array over the buffer. Host writes and kernel stores check it beside
`_writable` and refuse with a message that says what the array is and to copy
it. A copy is writable. DLPack exports of sealed memory are flagged read-only.
Writes Tack does not make (VTK writing memory it shares with Tack) are not
refused: they remain under the interop protocol, as today.

**The seal stops at DLPack.** The flag reaches only a consumer that asks for
DLPack 1.0 (`max_version`); the legacy protocol cannot carry it, so such a
consumer gets the memory as writable. Consumers that get the flag differ:
NumPy 2.4 makes a read-only array, while PyTorch 2.15 accepts the capsule
and ignores the flag, so a write through the tensor reaches the sealed memory
(checked 2026-11-16). No consumer refuses it. Exporting sealed structure stays
zero-copy; what a consumer then writes is its own violation, outside D1, for
D6's checks.

The cost is one attribute read per array a kernel may store to, per call, on
the dispatch fast path.

*Recommended.*

### D2. What is sealed, and when

Sealed when built, the structure Tack itself makes:

- **Topologies' stored arrays that Tack allocated**, at the end of their
  constructors: a topology a filter or reader makes (types, offsets,
  connectivity; polyhedra's face and cell arrays; polygons' loops; polylines'
  lines), and arrays a constructor converts or copies.
- **Derived structural arrays**, after their last construction write: faces
  and edges and their sides, point links, cell points, groups' rows, matched
  polygon facets, polyline ends, contour's sides, the boundary surface's
  cells and points.
- **Spaces' tables**: an `H1` space's checked DOF table and its order-2
  numbering, an `L2` space's orders and offsets.
- **An octree's tree** (`tree`, `level`, `key`), whose host copies the
  octree reads.
- **Implicit arrays** have no buffer; sealing one seals the stored arrays it
  holds.

**A caller's arrays are not sealed.** Constructors keep the arrays they are
given, without a copy, and in situ those are often a solver's own (its mesh
in Tack arrays, wrapped as a topology for analysis and remeshed or refined
in place between steps). A seal is one-way, so sealing them would lock the
solver out of its own memory for good, even after the topology is dropped,
and copying would cost the GPU memory in situ cannot spare. Instead:

- the topology holds its own **read-only view** of each array it is given
  (the per-array `_writable`), so no filter writes through it, while the
  caller's array object stays writable;
- the topology records the **revision** (`arrays._revision`) of the arrays it
  was given, and every structural cache it keeps checks it on use: one check
  for the topology, not one per cache. When it has moved, the topology drops
  its derived structure (sealed arrays included, which are rebuilt, not
  written) and rebuilds on demand, as D4's value caches do.

So the bug above gives the right answer instead of a refusal: after
`connectivity.fill(0)`, `extract_surface` rebuilds the faces. A store to the
derived faces' rows, which Tack made, is still refused. Writes the caller
makes outside Tack's view need D4's explicit call.

A write to derived structure is a **breaking change**, for the changelog,
though nothing in Tack, its tests or its guides makes one (the prototype).

*Recommended.*

### D3. Structural attributes cannot be reassigned

Today every structural array is a plain attribute (`topology.connectivity =
other` works, and leaves every cache stale), as are counts and dimensions
(`num_points`, `point_dims`). They become read-only properties over private
storage. `PolylineTopology.faces()` sets attributes after construction today
and moves to private storage behind the same properties.

*Recommended.*

### D4. Every cache derived from values tracks their revision

Values stay writable, so a cache derived from them must notice a write: an
automatic cache rebuilds, and a locator passed in explicitly that no longer
serves its dataset is refused (as `CellLocator.serves` does today). Already
tracked: the locators, `_mesh_kept`, the boundary surface's geometry,
serendipity's full values. To add:

- `Set` flags, derived from `Set.ids` (a plain public attribute today):
  writing a set's ids leaves its flags, and every filter that reads them,
  on the old members (the prototype confirmed it);
- the subgroups cached on a topology by `(group, spaces or sets)`, which
  depend on a set's membership;
- subdivision's crossings, carried by field identity (to check).

Not needed after all: the ghost exchange plans a partitioned dataset keeps
map ghost entries to their owners by index, so moving points or changing
values leaves them right; only a renumbering would not, and D2 rules that
out. An octree's host copies of its tree are structure, sealed under D2.

And one the other way: a locator's cell neighbours depend on structure only,
but are rebuilt whenever the values change; they move to the topology.

**Writes Tack cannot see** (a solver or a Catalyst adaptor writing positions
or fields through DLPack or `wrap_ptr`, every step) move no revision, so a
cached locator would serve the old points. Stage 1 adds an explicit call,
`Array.modified()` (VTK's `Modified()`), which counts a write to the array's
buffer as Tack's own writes are counted, so every revision-tracked cache, and
D2's topology check, notices. The in-situ documentation says to call it on
each array written outside Tack, each step. D6's checks are the safety net
for a forgotten call.

*Recommended.*

### D5. Sent arrays are read-only to the sender

`Blocks.exchange` documents that a message's arrays belong to the receiver
afterwards, but nothing enforces it, and a same-rank receiver gets the
sender's very object. Each sent array becomes read-only to its sender after
`exchange` returns, on every transport, and a same-rank receiver gets a new
array over the same memory with the sent array's writability. This is
ownership moving, not immutability, so it is the per-array flag
(`_writable`), not D1's buffer seal, which would lock the receiver too. Its
limit: views of the sent array made before sending can still write.

Reusing one send buffer step after step is the normal pattern for a
solver's halo exchange, and the hazard is narrower than the whole of
`exchange`: across ranks, `exchange` waits for every request before it
returns, so the receiver holds its own copy and the sender may write its
buffer again. Only a same-rank receiver (and the threaded test world) gets
the sender's memory. So `exchange` takes **`keep=True`**: the sender keeps
its arrays writable, and a same-rank receiver gets a copy instead of the
sender's memory. That is the supported pattern for reused send buffers, at
the cost of a copy on the same rank only.

Without `keep`, a sender that writes a sent array gets an error naming
`keep=True`: a **breaking change**. Nothing in Tack reuses one (`Plan.run`
gathers into a fresh array per send).

*Recommended.*

### D6. Checking external writes: later

Tack cannot see writes through memory it shares (VTK writing a connectivity
it handed Tack). A debug switch of its own (never `TACK_DEBUG`, whose suite
run must stay fast) could hash shared structural memory and check it at each
filter's entry. It catches violations of a protocol, not Tack's own writes,
which D1 refuses. Not needed for stage 1's safety.

*Recommended: defer to stage 2.*

## Not in stage 1

- Field and set mappings stay editable (`data.fields[name] = field`): 75 such
  assignments in tack-vis and over 600 in tests, docs and examples.
- Values are not locked: no leases, no generations, no `initialize()`,
  `snapshot()` or `with_fields`.
- `update_ghosts`, communication plans and AMR fills keep writing values in
  place; their writes are counted, so value caches see them.

## Later stages

Value immutability (a dataset published as a generation, values locked while
it is held, an explicit reset) waits for three things, each designed and
tested first:

1. **Caches without reference cycles.** A topology's cached locator holds a
   dataset, so a lock that ends when the dataset is freed would end when the
   cycle collector runs.
2. **A first-class in-situ loop.** `advance(u)` must not fail because the
   previous step's dataset still holds `u`.
3. **A rule for in-place distributed writes**: `update_ghosts`, `Plan.run`
   and AMR fills, which write published data on purpose.

## Prototype

Branch `experiment/mutation-rules` (`a26518cd`, local): D1 and D2 (the seal,
applied to topologies' stored arrays, their derived structure and an
octree's tree), D4 for sets and the subgroups by them, and D5 (the October
experiment, `e86415e7`). Not prototyped: D3's properties and D6. On `main`
at `4756ae3d`, the M1 Max CPU.

1. **Sealing breaks nothing in Tack.** The whole suite passes with structure
   sealed: 6859 passed, 372 skipped, no failures. So do the 164 guide
   examples. Nothing in Tack, its tests or its guides writes a topology's
   structure after building it; the refusal costs only code that does,
   outside.
2. **It costs about 18 ns per array a kernel may store to**, in the check on
   every call (364 against 419 ns for three such arrays, the best of nine
   runs on a loaded machine). A warm call is 7 to 10 µs, so under 1%.
3. **Zero-copy export still shares memory.** A topology with 64-bit ids
   exported to VTK with `copy=False` gives VTK the same memory as on `main`;
   VTK's array over it is writable to VTK, as before, under the interop
   protocol (D6). 32-bit ids are converted, on `main` too. A GPU export was
   not tried: it copies an array that is not `_writable`, and a sealed array
   still is, so it would share as today.
4. **Value caches.** Writing a set's ids left its flags on the old members
   (confirmed); keyed on the ids' revision, the flags and the subgroups by
   the set follow the write, and the reassigned ids too. The cell locator
   already notices moved points (a probe after moving every point finds
   nothing, as it should).
5. **Communication.** With sent arrays read-only to the sender, the
   communication, ghost update, ghost cells and partitioned tests pass, over
   threads and under `mpiexec -n 3`, staged and GPU-aware, but for the one
   test that asserted the receiver got the sender's very object, the
   intended change; it now checks that the receiver gets the same memory,
   writable, and the sender cannot write it.

The motivating bug is caught: after `topology.faces()`, a host write to the
connectivity and a kernel store to the faces' rows are both refused, with a
message to copy the array. **Verdict: stage 1 is viable** as planned, with D4
smaller than the inventory suggested.

## Open questions

- **When.** D2 and D5 refuse what worked before, which no deprecation can
  phase in. A release could first warn where it will later refuse, if the
  break should be softer than a changelog entry.
- **Sets.** Membership is treated as a value here (writable, tracked). Sealing
  a set's ids instead would be simpler, but forbids editing a set in place.
- **Geometry.** Stage 1 leaves positions writable, as every value. Whether
  some value (geometry first) should be sealed before stage 3 is open.
- **CuPy.** Whether it honours, ignores or refuses DLPack's read-only flag is
  unchecked (it needs a CUDA machine). D1 holds whichever it does.
- **Not prototyped:** D2's split between Tack's structure and a caller's
  (the prototype sealed the caller's arrays too), `Array.modified()` and
  `exchange(keep=True)`.

## Order

Each step is its own pull request, linking this plan:

1. D1 and D2, the seal and where it applies, the topology's check of the
   arrays it was given, and D3's properties.
2. D4, the value caches' revisions, with a test per cache, and
   `Array.modified()`.
3. D5, sent arrays, with `keep=True`.
4. This plan's status: done, with what changed on the way.
