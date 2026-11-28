### Changed

- On Metal, a host access of an array (`numpy()`, indexing, `fill`, an upload)
  waits only for the queued launches that use its memory, not for every launch
  queued: a thread's small uploads no longer wait for another thread's long
  launches, and a new array waits for nothing. A launch's error surfaces at a
  host access of an array it used, or at `tack.sync()`.
