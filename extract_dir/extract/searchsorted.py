@tf_export("searchsorted")
@dispatch.add_dispatch_support
def searchsorted(sorted_sequence,
                 values,
                 side="left",
                 out_type=dtypes.int32,
                 name=None):
  """Searches for where a value would go in a sorted sequence.
  This is not a method for checking containment (like python `in`).
  The typical use case for this operation is "binning", "bucketing", or
  "discretizing". The `values` are assigned to bucket-indices based on the
  **edges** listed in `sorted_sequence`. This operation
  returns the bucket-index for each value.
  >>> edges = [-1, 3.3, 9.1, 10.0]
  >>> values = [0.0, 4.1, 12.0]
  >>> tf.searchsorted(edges, values).numpy()
  array([1, 2, 4], dtype=int32)
  The `side` argument controls which index is returned if a value lands exactly
  on an edge:
  >>> seq = [0, 3, 9, 10, 10]
  >>> values = [0, 4, 10]
  >>> tf.searchsorted(seq, values).numpy()
  array([0, 2, 3], dtype=int32)
  >>> tf.searchsorted(seq, values, side="right").numpy()
  array([1, 2, 5], dtype=int32)
  The `axis` is not settable for this operation. It always operates on the
  innermost dimension (`axis=-1`). The operation will accept any number of
  outer dimensions. Here it is applied to the rows of a matrix:
  >>> sorted_sequence = [[0., 3., 8., 9., 10.],
  ...                    [1., 2., 3., 4., 5.]]
  >>> values = [[9.8, 2.1, 4.3],
  ...           [0.1, 6.6, 4.5, ]]
  >>> tf.searchsorted(sorted_sequence, values).numpy()
  array([[4, 1, 2],
         [0, 5, 4]], dtype=int32)
  Note: This operation assumes that `sorted_sequence` **is sorted** along the
  innermost axis, maybe using `tf.sort(..., axis=-1)`. **If the sequence is not
  sorted, no error is raised** and the content of the returned tensor is not well
  defined.
  Args:
    sorted_sequence: N-D `Tensor` containing a sorted sequence.
    values: N-D `Tensor` containing the search values.
    side: 'left' or 'right'; 'left' corresponds to lower_bound and 'right' to
      upper_bound.
    out_type: The output type (`int32` or `int64`).  Default is `tf.int32`.
    name: Optional name for the operation.
  Returns:
    An N-D `Tensor` the size of `values` containing the result of applying
    either lower_bound or upper_bound (depending on side) to each value.  The
    result is not a global index to the entire `Tensor`, but the index in the
    last dimension.
  Raises:
    ValueError: If the last dimension of `sorted_sequence >= 2^31-1` elements.
                If the total size of `values` exceeds `2^31 - 1` elements.
                If the first `N-1` dimensions of the two tensors don't match.
  """
  sequence_size = shape_internal(sorted_sequence)[-1]
  values_size = shape_internal(values)[-1]
  sorted_sequence_2d = reshape(sorted_sequence, [-1, sequence_size])
  values_2d = reshape(values, [-1, values_size])
  if side == "right":
    output = gen_array_ops.upper_bound(sorted_sequence_2d, values_2d, out_type,
                                       name)
  elif side == "left":
    output = gen_array_ops.lower_bound(sorted_sequence_2d, values_2d, out_type,
                                       name)
  else:
    raise ValueError("Argument `side` must be either 'right' or 'left'. "
                     f"Received: `side` = '{side}'.")
  return reshape(output, shape_internal(values))
