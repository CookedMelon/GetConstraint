@tf_export("math.ceil", v1=["math.ceil", "ceil"])
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints("ceil")
def ceil(x, name=None):
  """Return the ceiling of the input, element-wise.
  For example:
  >>> tf.math.ceil([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
  <tf.Tensor: shape=(7,), dtype=float32,
  numpy=array([-1., -1., -0.,  1.,  2.,  2.,  2.], dtype=float32)>
  Args:
    x: A `tf.Tensor`. Must be one of the following types: `bfloat16`, `half`,
      `float32`, `float64`. `int32`
    name: A name for the operation (optional).
  Returns:
    A `tf.Tensor`. Has the same type as `x`.
  @compatibility(numpy)
  Equivalent to np.ceil
  @end_compatibility
  """
  return gen_math_ops.ceil(x, name)
