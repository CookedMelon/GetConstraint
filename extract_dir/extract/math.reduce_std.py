@tf_export("math.reduce_std")
@dispatch.add_dispatch_support
def reduce_std(input_tensor, axis=None, keepdims=False, name=None):
  """Computes the standard deviation of elements across dimensions of a tensor.
  Reduces `input_tensor` along the dimensions given in `axis`.
  Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
  of the entries in `axis`, which must be unique. If `keepdims` is true, the
  reduced dimensions are retained with length 1.
  If `axis` is None, all dimensions are reduced, and a
  tensor with a single element is returned.
  For example:
  >>> x = tf.constant([[1., 2.], [3., 4.]])
  >>> tf.math.reduce_std(x)
  <tf.Tensor: shape=(), dtype=float32, numpy=1.118034>
  >>> tf.math.reduce_std(x, 0)
  <tf.Tensor: shape=(2,), dtype=float32, numpy=array([1., 1.], dtype=float32)>
  >>> tf.math.reduce_std(x, 1)
  <tf.Tensor: shape=(2,), dtype=float32, numpy=array([0.5, 0.5], dtype=float32)>
  Args:
    input_tensor: The tensor to reduce. Should have real or complex type.
    axis: The dimensions to reduce. If `None` (the default), reduces all
      dimensions. Must be in the range `[-rank(input_tensor),
      rank(input_tensor))`.
    keepdims: If true, retains reduced dimensions with length 1.
    name: A name scope for the associated operations (optional).
  Returns:
    The reduced tensor, of the same dtype as the input_tensor. Note,  for
    `complex64` or `complex128` input, the returned `Tensor` will be of type
    `float32` or `float64`, respectively.
  @compatibility(numpy)
  Equivalent to np.std
  Please note `np.std` has a `dtype` parameter that could be used to specify the
  output type. By default this is `dtype=float64`. On the other hand,
  `tf.math.reduce_std` has aggressive type inference from `input_tensor`.
  @end_compatibility
  """
  name = name if name else "reduce_std"
  with ops.name_scope(name):
    input_tensor = ops.convert_to_tensor(input_tensor)
    variance = reduce_variance(input_tensor, axis=axis, keepdims=keepdims)
    return gen_math_ops.sqrt(variance)
