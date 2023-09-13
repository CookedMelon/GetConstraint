@tf_export("sparse.mask", v1=["sparse.mask", "sparse_mask"])
@deprecation.deprecated_endpoints("sparse_mask")
def sparse_mask(a, mask_indices, name=None):
  """Masks elements of `IndexedSlices`.
  Given an `IndexedSlices` instance `a`, returns another `IndexedSlices` that
  contains a subset of the slices of `a`. Only the slices at indices not
  specified in `mask_indices` are returned.
  This is useful when you need to extract a subset of slices in an
  `IndexedSlices` object.
  For example:
  ```python
  # `a` contains slices at indices [12, 26, 37, 45] from a large tensor
  # with shape [1000, 10]
  a.indices  # [12, 26, 37, 45]
  tf.shape(a.values)  # [4, 10]
  # `b` will be the subset of `a` slices at its second and third indices, so
  # we want to mask its first and last indices (which are at absolute
  # indices 12, 45)
  b = tf.sparse.mask(a, [12, 45])
  b.indices  # [26, 37]
  tf.shape(b.values)  # [2, 10]
  ```
  Args:
    a: An `IndexedSlices` instance.
    mask_indices: Indices of elements to mask.
    name: A name for the operation (optional).
  Returns:
    The masked `IndexedSlices` instance.
  """
  with ops.name_scope(name, "sparse_mask", [a, mask_indices]) as name:
    indices = a.indices
    out_indices, to_gather = gen_array_ops.list_diff(indices, mask_indices)
    out_values = gather(a.values, to_gather, name=name)
    return indexed_slices.IndexedSlices(out_values, out_indices, a.dense_shape)
