### Added

- `merge_pieces` merges pieces with fields of higher order and curved cells at
  their order: `H1` fields above order 1, `L2` fields (an order per cell too) and
  a curved `H1` geometry travel as each owned cell's nodes, placed through the
  merged mesh's own numbering, so a node pieces share is one value. A field in
  different spaces on different pieces is refused. They were refused (part of
  #328; `ghost_cells` still refuses them).
