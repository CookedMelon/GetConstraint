@tf_export("math.xdivy")
@dispatch.register_binary_elementwise_api
@dispatch.add_dispatch_support
def xdivy(x, y, name=None):
  """Computes `x / y`.
  Given `x` and `y`, computes `x / y`. This function safely returns
  zero when `x = 0`, no matter what the value of `y` is.
  Example:
  >>> tf.math.xdivy(1., 2.)
  <tf.Tensor: shape=(), dtype=float32, numpy=0.5>
  >>> tf.math.xdivy(0., 1.)
  <tf.Tensor: shape=(), dtype=float32, numpy=0.0>
  >>> tf.math.xdivy(0., 0.)
  <tf.Tensor: shape=(), dtype=float32, numpy=0.0>
  >>> tf.math.xdivy(1., 0.)
  <tf.Tensor: shape=(), dtype=float32, numpy=inf>
  Args:
    x: A `tf.Tensor` of type `half`, `float32`, `float64`, `complex64`,
      `complex128`
    y: A `tf.Tensor` of type `half`, `float32`, `float64`, `complex64`,
      `complex128`
    name: A name for the operation (optional).
  Returns:
    `x / y`.
  """
  with ops.name_scope(name, "xdivy", [x]):
    return gen_math_ops.xdivy(x, y)
