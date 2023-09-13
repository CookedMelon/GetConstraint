@tf_export("sparse.from_dense")
def from_dense(tensor, name=None):
  """Converts a dense tensor into a sparse tensor.
  Only elements not equal to zero will be present in the result. The resulting
  `SparseTensor` has the same dtype and shape as the input.
  >>> sp = tf.sparse.from_dense([0, 0, 3, 0, 1])
  >>> sp.shape.as_list()
  [5]
  >>> sp.values.numpy()
  array([3, 1], dtype=int32)
  >>> sp.indices.numpy()
  array([[2],
         [4]])
  Args:
    tensor: A dense `Tensor` to be converted to a `SparseTensor`.
    name: Optional name for the op.
  Returns:
    The `SparseTensor`.
  """
  with ops.name_scope(name, "dense_to_sparse"):
    tensor = ops.convert_to_tensor(tensor)
    indices = array_ops.where_v2(
        math_ops.not_equal(tensor, array_ops.zeros_like(tensor)))
    values = array_ops.gather_nd(tensor, indices)
    shape = array_ops.shape(tensor, out_type=dtypes.int64)
    return sparse_tensor.SparseTensor(indices, values, shape)
