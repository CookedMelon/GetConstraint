@tf_export("nn.max_pool1d")
@dispatch.add_dispatch_support
def max_pool1d(input, ksize, strides, padding, data_format="NWC", name=None):
  """Performs the max pooling on the input.
  Note internally this op reshapes and uses the underlying 2d operation.
  Args:
    input: A 3-D `Tensor` of the format specified by `data_format`.
    ksize: An int or list of `ints` that has length `1` or `3`. The size of the
      window for each dimension of the input tensor.
    strides: An int or list of `ints` that has length `1` or `3`. The stride of
      the sliding window for each dimension of the input tensor.
    padding: Either the `string` `"SAME"` or `"VALID"` indicating the type of
      padding algorithm to use, or a list indicating the explicit paddings at
      the start and end of each dimension. See
      [here](https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2)
      for more information. When explicit padding is used and data_format is
      `"NWC"`, this should be in the form `[[0, 0], [pad_left, pad_right], [0,
      0]]`. When explicit padding used and data_format is `"NCW"`, this should
      be in the form `[[0, 0], [0, 0], [pad_left, pad_right]]`. When using
      explicit padding, the size of the paddings cannot be greater than the
      sliding window size.
    data_format: An optional string from: "NWC", "NCW". Defaults to "NWC".
    name: A name for the operation (optional).
  Returns:
    A `Tensor` of format specified by `data_format`.
    The max pooled output tensor.
  """
  with ops.name_scope(name, "MaxPool1d", [input]) as name:
    if isinstance(padding, (list, tuple)) and data_format == "NCHW_VECT_C":
      raise ValueError("`data_format='NCHW_VECT_C'` is not supported with "
                       f"explicit padding. Received: padding={padding}")
    if data_format is None:
      data_format = "NWC"
    channel_index = 1 if data_format.startswith("NC") else 2
    ksize = [1] + _get_sequence(ksize, 1, channel_index, "ksize")
    strides = [1] + _get_sequence(strides, 1, channel_index, "strides")
    padding, explicit_paddings = convert_padding(padding, 3)
    if padding == "EXPLICIT":
      explicit_paddings = [0, 0] + explicit_paddings
    expanding_dim = 1 if data_format == "NWC" else 2
    data_format = "NHWC" if data_format == "NWC" else "NCHW"
    input = array_ops.expand_dims_v2(input, expanding_dim)
    result = gen_nn_ops.max_pool(
        input,
        ksize=ksize,
        strides=strides,
        padding=padding,
        explicit_paddings=explicit_paddings,
        data_format=data_format,
        name=name)
    return array_ops.squeeze(result, expanding_dim)
