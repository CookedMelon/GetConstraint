@tf_export("sparse.reduce_max", v1=[])
def sparse_reduce_max_v2(
    sp_input, axis=None, keepdims=None, output_is_sparse=False, name=None):
  """Computes `tf.sparse.maximum` of elements across dimensions of a SparseTensor.
  This is the reduction operation for the elementwise `tf.sparse.maximum` op.
  This Op takes a SparseTensor and is the sparse counterpart to
  `tf.reduce_max()`.  In particular, this Op also returns a dense `Tensor`
  if `output_is_sparse` is `False`, or a `SparseTensor` if `output_is_sparse`
  is `True`.
  Note: A gradient is not defined for this function, so it can't be used
  in training models that need gradient descent.
  Reduces `sp_input` along the dimensions given in `axis`.  Unless
  `keepdims` is true, the rank of the tensor is reduced by 1 for each entry in
  `axis`. If `keepdims` is true, the reduced dimensions are retained
  with length 1.
  If `axis` has no entries, all dimensions are reduced, and a tensor
  with a single element is returned.  Additionally, the axes can be negative,
  similar to the indexing rules in Python.
  The values not defined in `sp_input` don't participate in the reduce max,
  as opposed to be implicitly assumed 0 -- hence it can return negative values
  for sparse `axis`. But, in case there are no values in
  `axis`, it will reduce to 0. See second example below.
  For example:
    # 'x' represents [[1, ?, 2]
    #                 [?, 3, ?]]
    # where ? is implicitly-zero.
    >>> x = tf.sparse.SparseTensor([[0, 0], [0, 2], [1, 1]], [1, 2, 3], [2, 3])
    >>> tf.sparse.reduce_max(x)
    <tf.Tensor: shape=(), dtype=int32, numpy=3>
    >>> tf.sparse.reduce_max(x, 0)
    <tf.Tensor: shape=(3,), dtype=int32, numpy=array([1, 3, 2], dtype=int32)>
    >>> tf.sparse.reduce_max(x, 1)
    <tf.Tensor: shape=(2,), dtype=int32, numpy=array([2, 3], dtype=int32)>
    >>> tf.sparse.reduce_max(x, 1, keepdims=True)
    <tf.Tensor: shape=(2, 1), dtype=int32, numpy=
    array([[2],
           [3]], dtype=int32)>
    >>> tf.sparse.reduce_max(x, [0, 1])
    <tf.Tensor: shape=(), dtype=int32, numpy=3>
    # 'y' represents [[-7, ?]
    #                 [ 4, 3]
    #                 [ ?, ?]
    >>> y = tf.sparse.SparseTensor([[0, 0,], [1, 0], [1, 1]], [-7, 4, 3],
    ... [3, 2])
    >>> tf.sparse.reduce_max(y, 1)
    <tf.Tensor: shape=(3,), dtype=int32, numpy=array([-7,  4,  0], dtype=int32)>
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
        gen_sparse_ops.sparse_reduce_max_sparse(
            sp_input.indices,
            sp_input.values,
            sp_input.dense_shape,
            math_ops._ReductionDims(sp_input, axis),
            keepdims,
            name=name))
    return sparse_tensor.SparseTensor(output_ind, output_val, output_shape)
  return gen_sparse_ops.sparse_reduce_max(
      sp_input.indices,
      sp_input.values,
      sp_input.dense_shape,
      math_ops._ReductionDims(sp_input, axis),
      keepdims,
      name=name)
