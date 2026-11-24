### Fixed

- A kernel whose arguments would pass the device's limit on their total size
  (2 KB on Level Zero's Intel GPUs, 4 KB on CUDA and HIP) reads its arrays'
  addresses from one table instead of taking each as an argument, so a
  kernel over many arrays builds on every GPU backend (#474).
