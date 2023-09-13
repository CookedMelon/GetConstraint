@tf_export(v1=["nn.dilation2d"])
@dispatch.add_dispatch_support
def dilation2d_v1(  # pylint: disable=missing-docstring
    input,  # pylint: disable=redefined-builtin
    filter=None,  # pylint: disable=redefined-builtin
    strides=None,
    rates=None,
    padding=None,
    name=None,
    filters=None,
    dilations=None):
  filter = deprecated_argument_lookup("filters", filters, "filter", filter)
  rates = deprecated_argument_lookup("dilations", dilations, "rates", rates)
  return gen_nn_ops.dilation2d(input, filter, strides, rates, padding, name)
