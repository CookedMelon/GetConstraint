@tf_export(v1=["nn.conv3d"])
@dispatch.add_dispatch_support
def conv3d_v1(  # pylint: disable=missing-docstring,dangerous-default-value
    input,  # pylint: disable=redefined-builtin
    filter=None,  # pylint: disable=redefined-builtin
    strides=None,
    padding=None,
    data_format="NDHWC",
    dilations=[1, 1, 1, 1, 1],
    name=None,
    filters=None):
  filter = deprecated_argument_lookup("filters", filters, "filter", filter)
  return gen_nn_ops.conv3d(
      input, filter, strides, padding, data_format, dilations, name)
