@tf_export("math.reciprocal_no_nan")
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def reciprocal_no_nan(x, name=None):
  """Performs a safe reciprocal operation, element wise.
  If a particular element is zero, the reciprocal for that element is
  also set to zero.
  For example:
  ```python
  x = tf.constant([2.0, 0.5, 0, 1], dtype=tf.float32)
  tf.math.reciprocal_no_nan(x)  # [ 0.5, 2, 0.0, 1.0 ]
  ```
  Args:
    x: A `Tensor` of type `float16`, `float32`, `float64` `complex64` or
      `complex128`.
    name: A name for the operation (optional).
  Returns:
    A `Tensor` of same shape and type as `x`.
  Raises:
    TypeError: x must be of a valid dtype.
  """
  with ops.name_scope(name, "reciprocal_no_nan", [x]) as scope:
    x = ops.convert_to_tensor(x, name="x")
    one = constant_op.constant(1, dtype=x.dtype.base_dtype, name="one")
    return gen_math_ops.div_no_nan(one, x, name=scope)
