### Changed

- `CellLocator` of an image (`UniformCoordinates`) or a rectilinear grid
  (`CartesianProduct`) builds nothing and finds a point's cell by arithmetic:
  its candidates from its index coordinates, or a binary search of each axis,
  each tested and ranked as the bins' candidates are, so `probe`, `advect`,
  `streamlines`, `voxelize` and resampling give the same cells, coordinates
  and values bit for bit. At 1M points on a 256^3 image (CPU, M1 Max) a warm
  `find` takes 17 ms against 113 ms, and the bins' 0.4 to 1 s build is gone.
  `streamlines` there starts 0.4 to 0.7 s sooner, with no bins to build, and
  `neighbors=True` changes nothing, as the search is as cheap as the walk.
  `levels=1` or `levels=2` still bins the cells; `levels` now defaults to
  `None` (#371).
