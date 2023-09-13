@tf_export("sets.size", v1=["sets.size", "sets.set_size"])
@dispatch.add_dispatch_support
def set_size(a, validate_indices=True):
  """Compute number of unique elements along last dimension of `a`.
  Args:
    a: `SparseTensor`, with indices sorted in row-major order.
    validate_indices: Whether to validate the order and range of sparse indices
      in `a`. Note that setting this to `false` allows for undefined behavior
      when calling this function with invalid indices.
  Returns:
    `int32` `Tensor` of set sizes. For `a` ranked `n`, this is a `Tensor` with
    rank `n-1`, and the same 1st `n-1` dimensions as `a`. Each value is the
    number of unique elements in the corresponding `[0...n-1]` dimension of `a`.
  Raises:
    TypeError: If `a` is an invalid types.
  """
  a = sparse_tensor.convert_to_tensor_or_sparse_tensor(a, name="a")
  if not isinstance(a, sparse_tensor.SparseTensor):
    raise TypeError("Expected `SparseTensor`, got %s." % a)
  if a.values.dtype.base_dtype not in _VALID_DTYPES:
    raise TypeError(
        f"Invalid dtype `{a.values.dtype}` not in supported dtypes: "
        f"`{_VALID_DTYPES}`.")
  # pylint: disable=protected-access
  return gen_set_ops.set_size(a.indices, a.values, a.dense_shape,
                              validate_indices)
