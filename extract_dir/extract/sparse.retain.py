@tf_export("sparse.retain", v1=["sparse.retain", "sparse_retain"])
@deprecation.deprecated_endpoints("sparse_retain")
def sparse_retain(sp_input, to_retain):
  """Retains specified non-empty values within a `SparseTensor`.
  For example, if `sp_input` has shape `[4, 5]` and 4 non-empty string values:
      [0, 1]: a
      [0, 3]: b
      [2, 0]: c
      [3, 1]: d
  and `to_retain = [True, False, False, True]`, then the output will
  be a `SparseTensor` of shape `[4, 5]` with 2 non-empty values:
      [0, 1]: a
      [3, 1]: d
  Args:
    sp_input: The input `SparseTensor` with `N` non-empty elements.
    to_retain: A bool vector of length `N` with `M` true values.
  Returns:
    A `SparseTensor` with the same shape as the input and `M` non-empty
    elements corresponding to the true positions in `to_retain`.
  Raises:
    TypeError: If `sp_input` is not a `SparseTensor`.
  """
  sp_input = _convert_to_sparse_tensor(sp_input)
  to_retain = ops.convert_to_tensor(to_retain)
  # Shape checking, if shape is known at graph construction time
  retain_shape = to_retain.get_shape()
  retain_shape.assert_has_rank(1)
  if sp_input.values.get_shape().dims is not None:
    sp_input.values.get_shape().dims[0].assert_is_compatible_with(
        tensor_shape.dimension_at_index(retain_shape, 0))
  where_true = array_ops.reshape(array_ops.where_v2(to_retain), [-1])
  new_indices = array_ops.gather(sp_input.indices, where_true)
  new_values = array_ops.gather(sp_input.values, where_true)
  return sparse_tensor.SparseTensor(new_indices, new_values,
                                    array_ops.identity(sp_input.dense_shape))
