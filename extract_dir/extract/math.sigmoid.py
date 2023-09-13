@tf_export("math.sigmoid", "nn.sigmoid", "sigmoid")
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def sigmoid(x, name=None):
  r"""Computes sigmoid of `x` element-wise.
  Formula for calculating $\mathrm{sigmoid}(x) = y = 1 / (1 + \exp(-x))$.
  For $x \in (-\infty, \infty)$, $\mathrm{sigmoid}(x) \in (0, 1)$.
  Example Usage:
  If a positive number is large, then its sigmoid will approach to 1 since the
  formula will be `y = <large_num> / (1 + <large_num>)`
  >>> x = tf.constant([0.0, 1.0, 50.0, 100.0])
  >>> tf.math.sigmoid(x)
  <tf.Tensor: shape=(4,), dtype=float32,
  numpy=array([0.5, 0.7310586, 1.0, 1.0], dtype=float32)>
  If a negative number is large, its sigmoid will approach to 0 since the
  formula will be `y = 1 / (1 + <large_num>)`
  >>> x = tf.constant([-100.0, -50.0, -1.0, 0.0])
  >>> tf.math.sigmoid(x)
  <tf.Tensor: shape=(4,), dtype=float32, numpy=
  array([0.0000000e+00, 1.9287499e-22, 2.6894143e-01, 0.5],
        dtype=float32)>
  Args:
    x: A Tensor with type `float16`, `float32`, `float64`, `complex64`, or
      `complex128`.
    name: A name for the operation (optional).
  Returns:
    A Tensor with the same type as `x`.
  Usage Example:
  >>> x = tf.constant([-128.0, 0.0, 128.0], dtype=tf.float32)
  >>> tf.sigmoid(x)
  <tf.Tensor: shape=(3,), dtype=float32,
  numpy=array([0. , 0.5, 1. ], dtype=float32)>
  @compatibility(scipy)
  Equivalent to scipy.special.expit
  @end_compatibility
  """
  with ops.name_scope(name, "Sigmoid", [x]) as name:
    x = ops.convert_to_tensor(x, name="x")
    return gen_math_ops.sigmoid(x, name=name)
