### Changed

- On Metal an array dropped while launches that use it are queued gives its
  buffer back to the pool at once, for a later array to take in queue order,
  as CUDA's and HIP's stream-ordered pools do; it used to wait for the
  launches' command buffer to complete, so a 5M-hexahedra clip still held its
  19 MiB point map while it allocated its output (#483).
