@tf_export("math.reduce_mean", "reduce_mean", v1=[])
@dispatch.add_dispatch_support
def reduce_mean(input_tensor, axis=None, keepdims=False, name=None):
  """Computes the mean of elements across dimensions of a tensor.
  Reduces `input_tensor` along the dimensions given in `axis` by computing the
  mean of elements across the dimensions in `axis`.
  Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
  of the entries in `axis`, which must be unique. If `keepdims` is true, the
  reduced dimensions are retained with length 1.
  If `axis` is None, all dimensions are reduced, and a tensor with a single
  element is returned.
  For example:
  >>> x = tf.constant([[1., 1.], [2., 2.]])
  >>> tf.reduce_mean(x)
  <tf.Tensor: shape=(), dtype=float32, numpy=1.5>
  >>> tf.reduce_mean(x, 0)
  <tf.Tensor: shape=(2,), dtype=float32, numpy=array([1.5, 1.5], dtype=float32)>
  >>> tf.reduce_mean(x, 1)
  <tf.Tensor: shape=(2,), dtype=float32, numpy=array([1., 2.], dtype=float32)>
  Args:
    input_tensor: The tensor to reduce. Should have numeric type.
    axis: The dimensions to reduce. If `None` (the default), reduces all
      dimensions. Must be in the range `[-rank(input_tensor),
      rank(input_tensor))`.
    keepdims: If true, retains reduced dimensions with length 1.
    name: A name for the operation (optional).
  Returns:
    The reduced tensor.
  @compatibility(numpy)
  Equivalent to np.mean
  Please note that `np.mean` has a `dtype` parameter that could be used to
  specify the output type. By default this is `dtype=float64`. On the other
  hand, `tf.reduce_mean` has an aggressive type inference from `input_tensor`,
  for example:
  >>> x = tf.constant([1, 0, 1, 0])
  >>> tf.reduce_mean(x)
  <tf.Tensor: shape=(), dtype=int32, numpy=0>
  >>> y = tf.constant([1., 0., 1., 0.])
  >>> tf.reduce_mean(y)
  <tf.Tensor: shape=(), dtype=float32, numpy=0.5>
  @end_compatibility
  """
  keepdims = False if keepdims is None else bool(keepdims)
  return _may_reduce_to_scalar(
      keepdims, axis,
      gen_math_ops.mean(
          input_tensor, _ReductionDims(input_tensor, axis), keepdims,
          name=name))
