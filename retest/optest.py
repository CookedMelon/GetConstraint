

@tf_export("nn.dilation2d", v1=[])
@dispatch.add_dispatch_support
def dilation2d_v2(
    input,   # pylint: disable=redefined-builtin
    filters,  # pylint: disable=redefined-builtin
    strides,
    padding,
    data_format,
    dilations,
    name=None):
  """Computes the grayscale dilation of 4-D `input` and 3-D `filters` tensors.

  The `input` tensor has shape `[batch, in_height, in_width, depth]` and the
  `filters` tensor has shape `[filter_height, filter_width, depth]`, i.e., each
  input channel is processed independently of the others with its own
  structuring function. The `output` tensor has shape
  """
def fun2():
  pass