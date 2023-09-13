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
  `[batch, out_height, out_width, depth]`. The spatial dimensions of the output
  tensor depend on the `padding` algorithm. We currently only support the
  default "NHWC" `data_format`.
  In detail, the grayscale morphological 2-D dilation is the max-sum correlation
  (for consistency with `conv2d`, we use unmirrored filters):
      output[b, y, x, c] =
         max_{dy, dx} input[b,
                            strides[1] * y + rates[1] * dy,
                            strides[2] * x + rates[2] * dx,
                            c] +
                      filters[dy, dx, c]
  Max-pooling is a special case when the filter has size equal to the pooling
  kernel size and contains all zeros.
  Note on duality: The dilation of `input` by the `filters` is equal to the
  negation of the erosion of `-input` by the reflected `filters`.
  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`,
      `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`,
      `uint32`, `uint64`.
      4-D with shape `[batch, in_height, in_width, depth]`.
    filters: A `Tensor`. Must have the same type as `input`.
      3-D with shape `[filter_height, filter_width, depth]`.
    strides: A list of `ints` that has length `>= 4`.
      The stride of the sliding window for each dimension of the input
      tensor. Must be: `[1, stride_height, stride_width, 1]`.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use. See
      [here](https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2)
      for more information.
    data_format: A `string`, only `"NHWC"` is currently supported.
    dilations: A list of `ints` that has length `>= 4`.
      The input stride for atrous morphological dilation. Must be:
      `[1, rate_height, rate_width, 1]`.
    name: A name for the operation (optional).
  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  if data_format != "NHWC":
    raise ValueError("`data_format` values other  than 'NHWC' are not "
                     f"supported. Received: data_format={data_format}")
  return gen_nn_ops.dilation2d(input=input,
                               filter=filters,
                               strides=strides,
                               rates=dilations,
                               padding=padding,
                               name=name)
