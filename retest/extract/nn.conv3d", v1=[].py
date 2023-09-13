@tf_export("nn.conv3d", v1=[])
@dispatch.add_dispatch_support
def conv3d_v2(input,  # pylint: disable=redefined-builtin,missing-docstring
              filters,
              strides,
              padding,
              data_format="NDHWC",
              dilations=None,
              name=None):
  if dilations is None:
    dilations = [1, 1, 1, 1, 1]
  return _conv3d_expanded_batch(input, filters, strides, padding, data_format,
                                dilations, name)
