@tf_export("ragged.constant")
@dispatch.add_dispatch_support
def constant(pylist, dtype=None, ragged_rank=None, inner_shape=None,
             name=None, row_splits_dtype=dtypes.int64):
  """Constructs a constant RaggedTensor from a nested Python list.
  Example:
  >>> tf.ragged.constant([[1, 2], [3], [4, 5, 6]])
  <tf.RaggedTensor [[1, 2], [3], [4, 5, 6]]>
  All scalar values in `pylist` must have the same nesting depth `K`, and the
  returned `RaggedTensor` will have rank `K`.  If `pylist` contains no scalar
  values, then `K` is one greater than the maximum depth of empty lists in
  `pylist`.  All scalar values in `pylist` must be compatible with `dtype`.
  Args:
    pylist: A nested `list`, `tuple` or `np.ndarray`.  Any nested element that
      is not a `list`, `tuple` or `np.ndarray` must be a scalar value
      compatible with `dtype`.
    dtype: The type of elements for the returned `RaggedTensor`.  If not
      specified, then a default is chosen based on the scalar values in
      `pylist`.
    ragged_rank: An integer specifying the ragged rank of the returned
      `RaggedTensor`.  Must be nonnegative and less than `K`. Defaults to
      `max(0, K - 1)` if `inner_shape` is not specified.  Defaults to
      `max(0, K - 1 - len(inner_shape))` if `inner_shape` is specified.
    inner_shape: A tuple of integers specifying the shape for individual inner
      values in the returned `RaggedTensor`.  Defaults to `()` if `ragged_rank`
      is not specified.  If `ragged_rank` is specified, then a default is chosen
      based on the contents of `pylist`.
    name: A name prefix for the returned tensor (optional).
    row_splits_dtype: data type for the constructed `RaggedTensor`'s row_splits.
      One of `tf.int32` or `tf.int64`.
  Returns:
    A potentially ragged tensor with rank `K` and the specified `ragged_rank`,
    containing the values from `pylist`.
  Raises:
    ValueError: If the scalar values in `pylist` have inconsistent nesting
      depth; or if ragged_rank or inner_shape are incompatible with `pylist`.
  """
  def ragged_factory(values, row_splits):
    row_splits = constant_op.constant(row_splits, dtype=row_splits_dtype)
    return ragged_tensor.RaggedTensor.from_row_splits(values, row_splits,
                                                      validate=False)
  with ops.name_scope(name, "RaggedConstant"):
    return _constant_value(ragged_factory, constant_op.constant, pylist, dtype,
                           ragged_rank, inner_shape)
