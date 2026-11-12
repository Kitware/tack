### Fixed

- The CPU backend keeps a kernel serial once threading it lost to
  contention (every thread on one atomic counter) unless a later threaded
  run is faster by the wall clock too. On a busy host its threads took
  turns, each meeting less contention, so their own rates said threading
  had won when it had not, and the kernel threaded again for a while.
