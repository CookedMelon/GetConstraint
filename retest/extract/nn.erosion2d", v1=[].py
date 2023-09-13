@tf_export("nn.erosion2d", v1=[])
@dispatch.add_dispatch_support
def erosion2d_v2(value,
                 filters,
                 strides,
                 padding,
                 data_format,
                 dilations,
                 name=None):
  """Computes the grayscale erosion of 4-D `value` and 3-D `filters` tensors.
  The `value` tensor has shape `[batch, in_height, in_width, depth]` and the
  `filters` tensor has shape `[filters_height, filters_width, depth]`, i.e.,
  each input channel is processed independently of the others with its own
  structuring function. The `output` tensor has shape
  `[batch, out_height, out_width, depth]`. The spatial dimensions of the
  output tensor depend on the `padding` algorithm. We currently only support the
  default "NHWC" `data_format`.
  In detail, the grayscale morphological 2-D erosion is given by:
      output[b, y, x, c] =
         min_{dy, dx} value[b,
                            strides[1] * y - dilations[1] * dy,
                            strides[2] * x - dilations[2] * dx,
                            c] -
                      filters[dy, dx, c]
  Duality: The erosion of `value` by the `filters` is equal to the negation of
  the dilation of `-value` by the reflected `filters`.
  Args:
    value: A `Tensor`. 4-D with shape `[batch, in_height, in_width, depth]`.
    filters: A `Tensor`. Must have the same type as `value`.
      3-D with shape `[filters_height, filters_width, depth]`.
    strides: A list of `ints` that has length `>= 4`.
      1-D of length 4. The stride of the sliding window for each dimension of
      the input tensor. Must be: `[1, stride_height, stride_width, 1]`.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use. See
      [here](https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2)
      for more information.
    data_format: A `string`, only `"NHWC"` is currently supported.
    dilations: A list of `ints` that has length `>= 4`.
      1-D of length 4. The input stride for atrous morphological dilation.
      Must be: `[1, rate_height, rate_width, 1]`.
    name: A name for the operation (optional). If not specified "erosion2d"
      is used.
  Returns:
    A `Tensor`. Has the same type as `value`.
    4-D with shape `[batch, out_height, out_width, depth]`.
  Raises:
    ValueError: If the `value` depth does not match `filters`' shape, or if
      padding is other than `'VALID'` or `'SAME'`.
  """
  if data_format != "NHWC":
    raise ValueError("`data_format` values other  than 'NHWC' are not "
                     f"supported. Received: data_format={data_format}")
  with ops.name_scope(name, "erosion2d", [value, filters]) as name:
    # Reduce erosion to dilation by duality.
    return math_ops.negative(
        gen_nn_ops.dilation2d(
            input=math_ops.negative(value),
            filter=array_ops.reverse_v2(filters, [0, 1]),
            strides=strides,
            rates=dilations,
            padding=padding,
            name=name))
