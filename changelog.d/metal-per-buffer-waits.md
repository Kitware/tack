### Changed

- On Metal, a host access of an array (`numpy()`, indexing, `fill`, an upload)
  waits only for the queued launches that use its memory, not for every launch
  queued: a thread's small uploads no longer wait for another thread's long
  launches, and a new array waits for nothing. A launch's error surfaces at a
  host access of an array it used, or at `tack.sync()`.
- On Metal, each thread launches on a command queue of its own, so one thread's
  launches run beside another's instead of in line behind them; a launch that
  uses an array another thread's queued launches use waits for them first, and
  `tack.sync()` waits for every thread's. The buffer pool gives a thread a
  buffer another thread's queued launches still use only when it has no other.
