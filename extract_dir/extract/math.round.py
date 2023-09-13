@tf_export("math.round", "round")
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def round(x, name=None):  # pylint: disable=redefined-builtin
  """Rounds the values of a tensor to the nearest integer, element-wise.
  Rounds half to even.  Also known as bankers rounding. If you want to round
  according to the current system rounding mode use tf::cint.
  For example:
  ```python
  x = tf.constant([0.9, 2.5, 2.3, 1.5, -4.5])
  tf.round(x)  # [ 1.0, 2.0, 2.0, 2.0, -4.0 ]
  ```
  Args:
    x: A `Tensor` of type `float16`, `float32`, `float64`, `int32`, or `int64`.
    name: A name for the operation (optional).
  Returns:
    A `Tensor` of same shape and type as `x`.
  """
  x = ops.convert_to_tensor(x, name="x")
  if x.dtype.is_integer:
    return x
  else:
    return gen_math_ops.round(x, name=name)
