@tf_export("sparse.maximum", v1=["sparse.maximum", "sparse_maximum"])
@deprecation.deprecated_endpoints("sparse_maximum")
def sparse_maximum(sp_a, sp_b, name=None):
  """Returns the element-wise max of two SparseTensors.
  Assumes the two SparseTensors have the same shape, i.e., no broadcasting.
  Example:
    >>> sp_zero = tf.sparse.SparseTensor([[0]], [0], [7])
    >>> sp_one = tf.sparse.SparseTensor([[1]], [1], [7])
    >>> res = tf.sparse.maximum(sp_zero, sp_one)
    >>> res.indices
    <tf.Tensor: shape=(2, 1), dtype=int64, numpy=
    array([[0],
           [1]])>
    >>> res.values
    <tf.Tensor: shape=(2,), dtype=int32, numpy=array([0, 1], dtype=int32)>
    >>> res.dense_shape
    <tf.Tensor: shape=(1,), dtype=int64, numpy=array([7])>
  The reduction version of this elementwise operation is `tf.sparse.reduce_max`
  Args:
    sp_a: a `SparseTensor` operand whose dtype is real, and indices
      lexicographically ordered.
    sp_b: the other `SparseTensor` operand with the same requirements (and the
      same shape).
    name: optional name of the operation.
  Returns:
    output: the output SparseTensor.
  """
  with ops.name_scope(
      name, "SparseSparseMaximum",
      [sp_a.indices, sp_a.values, sp_b.indices, sp_b.values]) as name:
    out_indices, out_values = gen_sparse_ops.sparse_sparse_maximum(
        sp_a.indices,
        sp_a.values,
        sp_a.dense_shape,
        sp_b.indices,
        sp_b.values,
        sp_b.dense_shape,
        name=name)
  return sparse_tensor.SparseTensor(out_indices, out_values, sp_a.dense_shape)
