@tf_export("math.ndtri")
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def ndtri(x, name=None):
  """Compute quantile of Standard Normal.
  Args:
    x: `Tensor` with type `float` or `double`.
    name: A name for the operation (optional).
  Returns:
    Inverse error function of `x`.
  """
  with ops.name_scope(name, "ndtri", [x]):
    return gen_math_ops.ndtri(x)
