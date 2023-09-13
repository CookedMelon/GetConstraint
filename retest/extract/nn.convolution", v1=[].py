@tf_export("nn.convolution", v1=[])
@dispatch.add_dispatch_support
def convolution_v2(  # pylint: disable=missing-docstring
    input,  # pylint: disable=redefined-builtin
    filters,
    strides=None,
    padding="VALID",
    data_format=None,
    dilations=None,
    name=None):
  return convolution_internal(
      input,  # pylint: disable=redefined-builtin
      filters,
      strides=strides,
      padding=padding,
      data_format=data_format,
      dilations=dilations,
      name=name)
