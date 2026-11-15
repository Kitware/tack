### Fixed

- Integers at the 64-bit edges keep their values (#412, found by a code
  review): an implicit array's integer operand is typed from its least and
  greatest values (`abs` of the i64 minimum wrapped, so it came out 0 in i32,
  and a u64 value past 2**63 came out negative in i64); `tack.arange` of
  integers is made in an integer type -- u64 for a range past i64 without a
  `dtype` -- where NumPy's rounded it through float64 (2**63 and 2**63 + 1 one
  value), and refuses ends no type holds; `CountingArray` computes in its
  declared type on the host and in kernels (a u64 start past 2**63 overflowed
  `numpy()` and mixed signed and unsigned in a kernel); and
  `from_dlpack(capsule, copy=True)` copies a raw capsule instead of failing.
