@tf_export("ragged.row_splits_to_segment_ids")
@dispatch.add_dispatch_support
def row_splits_to_segment_ids(splits, name=None, out_type=None):
  """Generates the segmentation corresponding to a RaggedTensor `row_splits`.
  Returns an integer vector `segment_ids`, where `segment_ids[i] == j` if
  `splits[j] <= i < splits[j+1]`.  Example:
  >>> print(tf.ragged.row_splits_to_segment_ids([0, 3, 3, 5, 6, 9]))
   tf.Tensor([0 0 0 2 2 3 4 4 4], shape=(9,), dtype=int64)
  Args:
    splits: A sorted 1-D integer Tensor.  `splits[0]` must be zero.
    name: A name prefix for the returned tensor (optional).
    out_type: The dtype for the return value.  Defaults to `splits.dtype`,
      or `tf.int64` if `splits` does not have a dtype.
  Returns:
    A sorted 1-D integer Tensor, with `shape=[splits[-1]]`
  Raises:
    ValueError: If `splits` is invalid.
  """
  with ops.name_scope(name, "RaggedSplitsToSegmentIds", [splits]) as name:
    splits = ops.convert_to_tensor(
        splits, name="splits",
        preferred_dtype=dtypes.int64)
    if splits.dtype not in (dtypes.int32, dtypes.int64):
      raise ValueError("splits must have dtype int32 or int64")
    splits.shape.assert_has_rank(1)
    if tensor_shape.dimension_value(splits.shape[0]) == 0:
      raise ValueError("Invalid row_splits: []")
    if out_type is None:
      out_type = splits.dtype
    else:
      out_type = dtypes.as_dtype(out_type)
    row_lengths = splits[1:] - splits[:-1]
    nrows = array_ops.shape(splits, out_type=out_type)[-1] - 1
    indices = math_ops.range(nrows)
    return ragged_util.repeat(indices, repeats=row_lengths, axis=0)
