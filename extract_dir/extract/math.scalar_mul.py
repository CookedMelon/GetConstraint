@tf_export("math.scalar_mul", "scalar_mul", v1=[])
@dispatch.register_binary_elementwise_api
@dispatch.add_dispatch_support
@_set_doc(scalar_mul.__doc__)
def scalar_mul_v2(scalar, x, name=None):
  with ops.name_scope(name, "scalar_mul", [x]) as name:
    return scalar_mul(scalar, x, name)
