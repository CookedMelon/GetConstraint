@tf_export("sparse.reduce_sum", v1=[])
def sparse_reduce_sum_v2(
    sp_input, axis=None, keepdims=None, output_is_sparse=False, name=None):
  """Computes `tf.sparse.add` of elements across dimensions of a SparseTensor.
  This is the reduction operation for the elementwise `tf.sparse.add` op.
  This Op takes a SparseTensor and is the sparse counterpart to
  `tf.reduce_sum()`.  In particular, this Op also returns a dense `Tensor`
  if `output_is_sparse` is `False`, or a `SparseTensor` if `output_is_sparse`
  is `True`.
  Note: if `output_is_sparse` is True, a gradient is not defined for this
  function, so it can't be used in training models that need gradient descent.
  Reduces `sp_input` along the dimensions given in `axis`.  Unless `keepdims` is
  true, the rank of the tensor is reduced by 1 for each entry in `axis`. If
  `keepdims` is true, the reduced dimensions are retained with length 1.
  If `axis` has no entries, all dimensions are reduced, and a tensor
  with a single element is returned.  Additionally, the axes can be negative,
  similar to the indexing rules in Python.
  For example:
    # 'x' represents [[1, ?, 1]
    #                 [?, 1, ?]]
    # where ? is implicitly-zero.
    >>> x = tf.sparse.SparseTensor([[0, 0], [0, 2], [1, 1]], [1, 1, 1], [2, 3])
    >>> tf.sparse.reduce_sum(x)
    <tf.Tensor: shape=(), dtype=int32, numpy=3>
    >>> tf.sparse.reduce_sum(x, 0)
    <tf.Tensor: shape=(3,), dtype=int32, numpy=array([1, 1, 1], dtype=int32)>
    >>> tf.sparse.reduce_sum(x, 1)  # Can also use -1 as the axis
    <tf.Tensor: shape=(2,), dtype=int32, numpy=array([2, 1], dtype=int32)>
    >>> tf.sparse.reduce_sum(x, 1, keepdims=True)
    <tf.Tensor: shape=(2, 1), dtype=int32, numpy=
    array([[2],
           [1]], dtype=int32)>
    >>> tf.sparse.reduce_sum(x, [0, 1])
    <tf.Tensor: shape=(), dtype=int32, numpy=3>
  Args:
    sp_input: The SparseTensor to reduce. Should have numeric type.
    axis: The dimensions to reduce; list or scalar. If `None` (the
      default), reduces all dimensions.
    keepdims: If true, retain reduced dimensions with length 1.
    output_is_sparse: If true, returns a `SparseTensor` instead of a dense
      `Tensor` (the default).
    name: A name for the operation (optional).
  Returns:
    The reduced Tensor or the reduced SparseTensor if `output_is_sparse` is
    True.
  """
  if keepdims is None:
    keepdims = False
  if output_is_sparse:
    output_ind, output_val, output_shape = (
        gen_sparse_ops.sparse_reduce_sum_sparse(
            sp_input.indices,
            sp_input.values,
            sp_input.dense_shape,
            math_ops._ReductionDims(sp_input, axis),
            keepdims,
            name=name))
    return sparse_tensor.SparseTensor(output_ind, output_val, output_shape)
  return gen_sparse_ops.sparse_reduce_sum(
      sp_input.indices,
      sp_input.values,
      sp_input.dense_shape,
      math_ops._ReductionDims(sp_input, axis),
      keepdims,
      name=name)
