@tf_export("math.xlog1py")
@dispatch.register_binary_elementwise_api
@dispatch.add_dispatch_support
def xlog1py(x, y, name=None):
  r"""Compute x * log1p(y).
  Given `x` and `y`, compute `x * log1p(y)`. This function safely returns
  zero when `x = 0`, no matter what the value of `y` is.
  Example:
  >>> tf.math.xlog1py(0., 1.)
  <tf.Tensor: shape=(), dtype=float32, numpy=0.>
  >>> tf.math.xlog1py(1., 1.)
  <tf.Tensor: shape=(), dtype=float32, numpy=0.6931472>
  >>> tf.math.xlog1py(2., 2.)
  <tf.Tensor: shape=(), dtype=float32, numpy=2.1972246>
  >>> tf.math.xlog1py(0., -1.)
  <tf.Tensor: shape=(), dtype=float32, numpy=0.>
  Args:
    x: A `tf.Tensor` of type `half`, `float32`, `float64`, `complex64`,
      `complex128`
    y: A `tf.Tensor` of type `half`, `float32`, `float64`, `complex64`,
      `complex128`
    name: A name for the operation (optional).
  Returns:
    `x * log1p(y)`.
  @compatibility(scipy)
  Equivalent to scipy.special.xlog1py
  @end_compatibility
  """
  with ops.name_scope(name, "xlog1py", [x]):
    return gen_math_ops.xlog1py(x, y)
