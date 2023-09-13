@tf_export("math.erfinv")
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def erfinv(x, name=None):
  """Compute inverse error function.
  Given `x`, compute the inverse error function of `x`. This function
  is the inverse of `tf.math.erf`.
  Args:
    x: `Tensor` with type `float` or `double`.
    name: A name for the operation (optional).
  Returns:
    Inverse error function of `x`.
  """
  with ops.name_scope(name, "erfinv", [x]):
    return gen_math_ops.erfinv(x)
