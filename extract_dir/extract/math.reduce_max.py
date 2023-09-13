@tf_export("math.reduce_max", "reduce_max", v1=[])
@dispatch.add_dispatch_support
def reduce_max(input_tensor, axis=None, keepdims=False, name=None):
  """Computes `tf.math.maximum` of elements across dimensions of a tensor.
  This is the reduction operation for the elementwise `tf.math.maximum` op.
  Reduces `input_tensor` along the dimensions given in `axis`.
  Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
  of the entries in `axis`, which must be unique. If `keepdims` is true, the
  reduced dimensions are retained with length 1.
  If `axis` is None, all dimensions are reduced, and a
  tensor with a single element is returned.
  Usage example:
    >>> x = tf.constant([5, 1, 2, 4])
    >>> tf.reduce_max(x)
    <tf.Tensor: shape=(), dtype=int32, numpy=5>
    >>> x = tf.constant([-5, -1, -2, -4])
    >>> tf.reduce_max(x)
    <tf.Tensor: shape=(), dtype=int32, numpy=-1>
    >>> x = tf.constant([4, float('nan')])
    >>> tf.reduce_max(x)
    <tf.Tensor: shape=(), dtype=float32, numpy=nan>
    >>> x = tf.constant([float('nan'), float('nan')])
    >>> tf.reduce_max(x)
    <tf.Tensor: shape=(), dtype=float32, numpy=nan>
    >>> x = tf.constant([float('-inf'), float('inf')])
    >>> tf.reduce_max(x)
    <tf.Tensor: shape=(), dtype=float32, numpy=inf>
  See the numpy docs for `np.amax` and `np.nanmax` behavior.
  Args:
    input_tensor: The tensor to reduce. Should have real numeric type.
    axis: The dimensions to reduce. If `None` (the default), reduces all
      dimensions. Must be in the range `[-rank(input_tensor),
      rank(input_tensor))`.
    keepdims: If true, retains reduced dimensions with length 1.
    name: A name for the operation (optional).
  Returns:
    The reduced tensor.
  """
  return reduce_max_with_dims(input_tensor, axis, keepdims, name,
                              _ReductionDims(input_tensor, axis))
