@tf_export("sparse.to_dense", v1=["sparse.to_dense", "sparse_tensor_to_dense"])
@deprecation.deprecated_endpoints("sparse_tensor_to_dense")
def sparse_tensor_to_dense(sp_input,
                           default_value=None,
                           validate_indices=True,
                           name=None):
  """Converts a `SparseTensor` into a dense tensor.
  For this sparse tensor with three non-empty values:
  >>> sp_input = tf.sparse.SparseTensor(
  ...   dense_shape=[3, 5],
  ...   values=[7, 8, 9],
  ...   indices =[[0, 1],
  ...             [0, 3],
  ...             [2, 0]])
  The output will be a dense `[3, 5]` tensor with values:
  >>> tf.sparse.to_dense(sp_input).numpy()
  array([[0, 7, 0, 8, 0],
         [0, 0, 0, 0, 0],
         [9, 0, 0, 0, 0]], dtype=int32)
  Note: Indices must be without repeats.  This is only tested if
  `validate_indices` is `True`.
  Args:
    sp_input: The input `SparseTensor`.
    default_value: Scalar value to set for indices not specified in
      `sp_input`.  Defaults to zero.
    validate_indices: A boolean value.  If `True`, indices are checked to make
      sure they are sorted in lexicographic order and that there are no repeats.
    name: A name prefix for the returned tensors (optional).
  Returns:
    A dense tensor with shape `sp_input.dense_shape` and values specified by
    the non-empty values in `sp_input`. Indices not in `sp_input` are assigned
    `default_value`.
  Raises:
    TypeError: If `sp_input` is not a `SparseTensor`.
  """
  sp_input = _convert_to_sparse_tensor(sp_input)
  if default_value is None:
    default_value = array_ops.zeros([], dtype=sp_input.dtype)
  return gen_sparse_ops.sparse_to_dense(
      sp_input.indices,
      sp_input.dense_shape,
      sp_input.values,
      default_value=default_value,
      validate_indices=validate_indices,
      name=name)
