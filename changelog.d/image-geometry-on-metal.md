### Fixed

- Tack's VTK filters (`tack.interop.vtk.algorithms`) run on an image given a
  device without f64 (Metal): VTK keeps an image's origin and spacing in
  doubles, so its geometry came in as f64 and the move to the device failed.
  There the geometry is now taken in f32, relative to the origin an f32
  conversion takes, so an image far from zero keeps its precision.
