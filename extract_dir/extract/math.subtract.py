@tf_export("math.subtract", "subtract")
@dispatch.register_binary_elementwise_api
@dispatch.add_dispatch_support
def subtract(x, y, name=None):
  return gen_math_ops.sub(x, y, name)
