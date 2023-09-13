@tf_export("nn.avg_pool2d", v1=[])
@dispatch.add_dispatch_support
def avg_pool2d(input, ksize, strides, padding, data_format="NHWC", name=None):  # pylint: disable=redefined-builtin
  """Performs the average pooling on the input.
  Each entry in `output` is the mean of the corresponding size `ksize`
  window in `value`.
  Args:
    input: A 4-D `Tensor` of shape `[batch, height, width, channels]` and type
      `float32`, `float64`, `qint8`, `quint8`, or `qint32`.
    ksize: An int or list of `ints` that has length `1`, `2` or `4`. The size of
      the window for each dimension of the input tensor.
    strides: An int or list of `ints` that has length `1`, `2` or `4`. The
      stride of the sliding window for each dimension of the input tensor.
    padding: A string, either `'VALID'` or `'SAME'`. The padding algorithm. See
      [here](https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2)
      for more information.
    data_format: A string. 'NHWC' and 'NCHW' are supported.
    name: Optional name for the operation.
  Returns:
    A `Tensor` with the same type as `value`.  The average pooled output tensor.
  """
  with ops.name_scope(name, "AvgPool2D", [input]) as name:
    if data_format is None:
      data_format = "NHWC"
    channel_index = 1 if data_format.startswith("NC") else 3
    ksize = _get_sequence(ksize, 2, channel_index, "ksize")
    strides = _get_sequence(strides, 2, channel_index, "strides")
    return gen_nn_ops.avg_pool(
        input,
        ksize=ksize,
        strides=strides,
        padding=padding,
        data_format=data_format,
        name=name)
