@tf_export("strings.substr", v1=[])
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def substr_v2(input, pos, len, unit="BYTE", name=None):
  return gen_string_ops.substr(input, pos, len, unit=unit, name=name)
