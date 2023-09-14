@tf_export(
    "nn.depthwise_conv2d_backprop_input",
    v1=[
        "nn.depthwise_conv2d_native_backprop_input",
        "nn.depthwise_conv2d_backprop_input"
    ])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints("nn.depthwise_conv2d_native_backprop_input")
def depthwise_conv2d_native_backprop_input(  # pylint: disable=redefined-builtin,dangerous-default-value
    input_sizes,
    filter,
    out_backprop,
    strides,
    padding,
    data_format="NHWC",
    dilations=[1, 1, 1, 1],
    name=None):
  r"""Computes the gradients of depthwise convolution with respect to the input.
  Args:
    input_sizes: A `Tensor` of type `int32`. An integer vector representing the
      shape of `input`, based on `data_format`.  For example, if `data_format`
      is 'NHWC' then `input` is a 4-D `[batch, height, width, channels]` tensor.
    filter: A `Tensor`. Must be one of the following types: `half`, `bfloat16`,
      `float32`, `float64`. 4-D with shape `[filter_height, filter_width,
      in_channels, depthwise_multiplier]`.
    out_backprop: A `Tensor`. Must have the same type as `filter`. 4-D with
      shape  based on `data_format`. For example, if `data_format` is 'NHWC'
      then out_backprop shape is `[batch, out_height, out_width, out_channels]`.
      Gradients w.r.t. the output of the convolution.
    strides: A list of `ints`. The stride of the sliding window for each
      dimension of the input of the convolution.
    padding: Controls how to pad the image before applying the convolution. Can
      be the string `"SAME"` or `"VALID"` indicating the type of padding
      algorithm to use, or a list indicating the explicit paddings at the start
      and end of each dimension. See
      [here](https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2)
      for more information. When explicit padding is used and data_format is
      `"NHWC"`, this should be in the form `[[0, 0], [pad_top, pad_bottom],
      [pad_left, pad_right], [0, 0]]`. When explicit padding used and
      data_format is `"NCHW"`, this should be in the form `[[0, 0], [0, 0],
      [pad_top, pad_bottom], [pad_left, pad_right]]`.
    data_format: An optional `string` from: `"NHWC", "NCHW"`. Defaults to
      `"NHWC"`. Specify the data format of the input and output data. With the
      default format "NHWC", the data is stored in the order of: [batch, height,
        width, channels].
      Alternatively, the format could be "NCHW", the data storage order of:
        [batch, channels, height, width].
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`. 1-D
      tensor of length 4.  The dilation factor for each dimension of `input`. If
      set to k > 1, there will be k-1 skipped cells between each filter element
      on that dimension. The dimension order is determined by the value of
      `data_format`, see above for details. Dilations in the batch and depth
      dimensions must be 1.
    name: A name for the operation (optional).
  Returns:
    A `Tensor`. Has the same type as `filter`.
  """
  padding, explicit_paddings = convert_padding(padding)
  return gen_nn_ops.depthwise_conv2d_native_backprop_input(
      input_sizes,
      filter,
      out_backprop,
      strides,
      padding,
      explicit_paddings=explicit_paddings,
      data_format=data_format,
      dilations=dilations,
      name=name)
