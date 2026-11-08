### Changed

- Cold compiles verify the IR twice, after lowering and before code
  generation, rather than at every pass. The final check repeats every
  earlier one but the GPU packing's, so a pass that breaks an invariant
  still stops before code generation. The stages between run in debug mode
  or with `TACK_VERIFY_ALL` set, as the test suite sets it, and then name the
  pass. On AhmedML's analysis sweep (244 kernels, cold, CPU) verification went
  from 2.6 s to 1.1 s, and compiling from 15.5 s to 13.8 s (#322).
