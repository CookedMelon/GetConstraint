@tf_export("math.acos", "acos")
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def acos(x, name=None):
  """Computes acos of x element-wise.
  Provided an input tensor, the `tf.math.acos` operation
  returns the inverse cosine of each element of the tensor.
  If `y = tf.math.cos(x)` then, `x = tf.math.acos(y)`.
  Input range is `[-1, 1]` and the output has a range of `[0, pi]`.
  For example:
  >>> x = tf.constant([1.0, -0.5, 3.4, 0.2, 0.0, -2], dtype = tf.float32)
  >>> tf.math.acos(x)
  <tf.Tensor: shape=(6,), dtype=float32,
  numpy= array([0. , 2.0943952, nan, 1.3694383, 1.5707964, nan],
  dtype=float32)>
  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`,
      `float32`, `float64`, `complex64`, `complex128`.
    name: A name for the operation (optional).
  Returns:
    A `Tensor`. Has the same type as x.
  """
  return gen_math_ops.acos(x, name)
