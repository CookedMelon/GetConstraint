@tf_export("nn.max_pool3d")
@dispatch.add_dispatch_support
def max_pool3d(input, ksize, strides, padding, data_format="NDHWC", name=None):
  """Performs the max pooling on the input.
  Args:
    input: A 5-D `Tensor` of the format specified by `data_format`.
    ksize: An int or list of `ints` that has length `1`, `3` or `5`. The size of
      the window for each dimension of the input tensor.
    strides: An int or list of `ints` that has length `1`, `3` or `5`. The
      stride of the sliding window for each dimension of the input tensor.
    padding: A string, either `'VALID'` or `'SAME'`. The padding algorithm. See
      [here](https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2)
      for more information.
    data_format: An optional string from: "NDHWC", "NCDHW". Defaults to "NDHWC".
      The data format of the input and output data. With the default format
      "NDHWC", the data is stored in the order of: [batch, in_depth, in_height,
        in_width, in_channels]. Alternatively, the format could be "NCDHW", the
      data storage order is: [batch, in_channels, in_depth, in_height,
        in_width].
    name: A name for the operation (optional).
  Returns:
    A `Tensor` of format specified by `data_format`.
    The max pooled output tensor.
  """
  with ops.name_scope(name, "MaxPool3D", [input]) as name:
    if data_format is None:
      data_format = "NDHWC"
    channel_index = 1 if data_format.startswith("NC") else 4
    ksize = _get_sequence(ksize, 3, channel_index, "ksize")
    strides = _get_sequence(strides, 3, channel_index, "strides")
    return gen_nn_ops.max_pool3d(
        input,
        ksize=ksize,
        strides=strides,
        padding=padding,
        data_format=data_format,
        name=name)
