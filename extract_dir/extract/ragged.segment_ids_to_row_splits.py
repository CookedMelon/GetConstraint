@tf_export("ragged.segment_ids_to_row_splits")
@dispatch.add_dispatch_support
def segment_ids_to_row_splits(segment_ids, num_segments=None,
                              out_type=None, name=None):
  """Generates the RaggedTensor `row_splits` corresponding to a segmentation.
  Returns an integer vector `splits`, where `splits[0] = 0` and
  `splits[i] = splits[i-1] + count(segment_ids==i)`.  Example:
  >>> print(tf.ragged.segment_ids_to_row_splits([0, 0, 0, 2, 2, 3, 4, 4, 4]))
  tf.Tensor([0 3 3 5 6 9], shape=(6,), dtype=int64)
  Args:
    segment_ids: A 1-D integer Tensor.
    num_segments: A scalar integer indicating the number of segments.  Defaults
      to `max(segment_ids) + 1` (or zero if `segment_ids` is empty).
    out_type: The dtype for the return value.  Defaults to `segment_ids.dtype`,
      or `tf.int64` if `segment_ids` does not have a dtype.
    name: A name prefix for the returned tensor (optional).
  Returns:
    A sorted 1-D integer Tensor, with `shape=[num_segments + 1]`.
  """
  # Local import bincount_ops to avoid import-cycle.
  from tensorflow.python.ops import bincount_ops  # pylint: disable=g-import-not-at-top
  if out_type is None:
    if isinstance(segment_ids, ops.Tensor):
      out_type = segment_ids.dtype
    elif isinstance(num_segments, ops.Tensor):
      out_type = num_segments.dtype
    else:
      out_type = dtypes.int64
  else:
    out_type = dtypes.as_dtype(out_type)
  with ops.name_scope(name, "SegmentIdsToRaggedSplits", [segment_ids]) as name:
    # Note: we cast int64 tensors to int32, since bincount currently only
    # supports int32 inputs.
    segment_ids = ragged_util.convert_to_int_tensor(segment_ids, "segment_ids",
                                                    dtype=dtypes.int32)
    segment_ids.shape.assert_has_rank(1)
    if num_segments is not None:
      num_segments = ragged_util.convert_to_int_tensor(num_segments,
                                                       "num_segments",
                                                       dtype=dtypes.int32)
      num_segments.shape.assert_has_rank(0)
    row_lengths = bincount_ops.bincount(
        segment_ids,
        minlength=num_segments,
        maxlength=num_segments,
        dtype=out_type)
    splits = array_ops.concat([[0], math_ops.cumsum(row_lengths)], axis=0)
    # Update shape information, if possible.
    if num_segments is not None:
      const_num_segments = tensor_util.constant_value(num_segments)
      if const_num_segments is not None:
        splits.set_shape(tensor_shape.TensorShape([const_num_segments + 1]))
    return splits
