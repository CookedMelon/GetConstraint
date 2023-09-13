@tf_export("sparse.map_values", v1=[])
@dispatch.add_dispatch_support
def map_values(op, *args, **kwargs):
  """Applies `op` to the `.values` tensor of one or more `SparseTensor`s.
  Replaces any `SparseTensor` in `args` or `kwargs` with its `values`
  tensor (which contains the non-default values for the SparseTensor),
  and then calls `op`.  Returns a `SparseTensor` that is constructed
  from the input `SparseTensor`s' `indices`, `dense_shape`, and the
  value returned by the `op`.
  If the input arguments contain multiple `SparseTensor`s, then they must have
  equal `indices` and dense shapes.
  Examples:
  >>> s = tf.sparse.from_dense([[1, 2, 0],
  ...                           [0, 4, 0],
  ...                           [1, 0, 0]])
  >>> tf.sparse.to_dense(tf.sparse.map_values(tf.ones_like, s)).numpy()
  array([[1, 1, 0],
         [0, 1, 0],
         [1, 0, 0]], dtype=int32)
  >>> tf.sparse.to_dense(tf.sparse.map_values(tf.multiply, s, s)).numpy()
  array([[ 1,  4,  0],
         [ 0, 16,  0],
         [ 1,  0,  0]], dtype=int32)
  >>> tf.sparse.to_dense(tf.sparse.map_values(tf.add, s, 5)).numpy()
  array([[6, 7, 0],
         [0, 9, 0],
         [6, 0, 0]], dtype=int32)
  Note: even though `tf.add(0, 5) != 0`, implicit zeros
  will remain unchanged. However, if the sparse tensor contains any explicit
  zeros, these will be affected by the mapping!
  Args:
    op: The operation that should be applied to the SparseTensor `values`. `op`
      is typically an element-wise operation (such as math_ops.add), but any
      operation that preserves the shape can be used.
    *args: Arguments for `op`.
    **kwargs: Keyword arguments for `op`.
  Returns:
    A `SparseTensor` whose `indices` and `dense_shape` matches the `indices`
    and `dense_shape` of all input `SparseTensor`s.
  Raises:
    ValueError: If args contains no `SparseTensor`, or if the `indices`
      or `dense_shape`s of the input `SparseTensor`s are not equal.
  """
  sparse_list = []
  inner_args = _replace_sparse_with_values(args, sparse_list)
  inner_kwargs = _replace_sparse_with_values(kwargs, sparse_list)
  if not sparse_list:
    raise ValueError("No SparseTensor in argument list of map_values")
  with ops.control_dependencies(_assert_sparse_compatible(sparse_list)):
    # Delegate to op, and then compose the result from the transformed values
    # and the known indices/dense shape. Since we ensure that indices and shape
    # are identical, we can just use the first one.
    return sparse_tensor.SparseTensor(sparse_list[0].indices,
                                      op(*inner_args, **inner_kwargs),
                                      sparse_list[0].dense_shape)
