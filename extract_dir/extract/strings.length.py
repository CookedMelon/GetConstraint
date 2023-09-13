@tf_export("strings.length", v1=[])
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def string_length_v2(input, unit="BYTE", name=None):
  return gen_string_ops.string_length(input, unit=unit, name=name)
