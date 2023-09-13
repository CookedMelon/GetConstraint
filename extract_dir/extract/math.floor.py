@tf_export("math.floor", "floor")
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def floor(x, name=None):
  """Returns element-wise largest integer not greater than x.
  Both input range is `(-inf, inf)` and the
  output range consists of all integer values.
  For example:
  >>> x = tf.constant([1.3324, -1.5, 5.555, -2.532, 0.99, float("inf")])
  >>> tf.floor(x).numpy()
  array([ 1., -2.,  5., -3.,  0., inf], dtype=float32)
  Args:
    x:  A `Tensor`. Must be one of the following types: `bfloat16`, `half`,
      `float32`, `float64`.
    name: A name for the operation (optional).
  Returns:
    A `Tensor`. Has the same type as x.
  """
  return gen_math_ops.floor(x, name)
