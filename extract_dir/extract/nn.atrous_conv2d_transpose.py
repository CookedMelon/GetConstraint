@tf_export("nn.atrous_conv2d_transpose")
@dispatch.add_dispatch_support
def atrous_conv2d_transpose(value,
                            filters,
                            output_shape,
                            rate,
                            padding,
                            name=None):
  """The transpose of `atrous_conv2d`.
  This operation is sometimes called "deconvolution" after
  (Zeiler et al., 2010), but is really the transpose (gradient) of
  `atrous_conv2d` rather than an actual deconvolution.
  Args:
    value: A 4-D `Tensor` of type `float`. It needs to be in the default `NHWC`
      format. Its shape is `[batch, in_height, in_width, in_channels]`.
    filters: A 4-D `Tensor` with the same type as `value` and shape
      `[filter_height, filter_width, out_channels, in_channels]`. `filters`'
      `in_channels` dimension must match that of `value`. Atrous convolution is
      equivalent to standard convolution with upsampled filters with effective
      height `filter_height + (filter_height - 1) * (rate - 1)` and effective
      width `filter_width + (filter_width - 1) * (rate - 1)`, produced by
      inserting `rate - 1` zeros along consecutive elements across the
      `filters`' spatial dimensions.
    output_shape: A 1-D `Tensor` of shape representing the output shape of the
      deconvolution op, of form `[batch, out_height, out_width, out_channels]`.
    rate: A positive int32. The stride with which we sample input values across
      the `height` and `width` dimensions. Equivalently, the rate by which we
      upsample the filter values by inserting zeros across the `height` and
      `width` dimensions. In the literature, the same parameter is sometimes
      called `input stride` or `dilation`.
    padding: A string, either `'VALID'` or `'SAME'`. The padding algorithm. See
      [here](https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2)
      for more information.
    name: Optional name for the returned tensor.
  Returns:
    A `Tensor` with the same type as `value`.
  Raises:
    ValueError: If input/output depth does not match `filters`' shape, or if
      padding is other than `'VALID'` or `'SAME'`, or if the `rate` is less
      than one, or if the output_shape is not a tensor with 4 elements.
  References:
    Deconvolutional Networks:
      [Zeiler et al., 2010]
      (https://ieeexplore.ieee.org/abstract/document/5539957)
      ([pdf]
      (http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.232.4023&rep=rep1&type=pdf))
  """
  with ops.name_scope(name, "atrous_conv2d_transpose",
                      [value, filters, output_shape]) as name:
    value = ops.convert_to_tensor(value, name="value")
    filters = ops.convert_to_tensor(filters, name="filters")
    if not value.get_shape().dims[3].is_compatible_with(filters.get_shape()[3]):
      raise ValueError(
          "`value` channel count must be compatible with `filters` input "
          f"channel count. Received: value.shape={value.get_shape()} with "
          f"channel count {value.get_shape()[3]} and "
          f"filters.shape={filters.get_shape()} with input channel count "
          f"{filters.get_shape()[3]}.")
    if rate < 1:
      raise ValueError(f"`rate` cannot be less than one. Received: rate={rate}")
    if rate == 1:
      return conv2d_transpose(
          value,
          filters,
          output_shape,
          strides=[1, 1, 1, 1],
          padding=padding,
          data_format="NHWC")
    output_shape_ = ops.convert_to_tensor(output_shape, name="output_shape")
    if not output_shape_.get_shape().is_compatible_with(
        tensor_shape.TensorShape([4])):
      raise ValueError("`output_shape` must have shape (4,). "
                       f"Received: output_shape={output_shape_.get_shape()}")
    if isinstance(output_shape, tuple):
      output_shape = list(output_shape)
    if isinstance(output_shape, (list, np.ndarray)):
      # output_shape's shape should be == [4] if reached this point.
      if not filters.get_shape().dims[2].is_compatible_with(output_shape[3]):
        raise ValueError(
            "`output_shape` channel count must be compatible with `filters` "
            f"output channel count. Received: output_shape={output_shape} with "
            f"channel count {output_shape[3]} and "
            f"filters.shape={filters.get_shape()} with output channel count "
            f"{filters.get_shape()[3]}.")
    # We have two padding contributions. The first is used for converting "SAME"
    # to "VALID". The second is required so that the height and width of the
    # zero-padded value tensor are multiples of rate.
    # Padding required to reduce to "VALID" convolution
    if padding == "SAME":
      # Handle filters whose shape is unknown during graph creation.
      if filters.get_shape().is_fully_defined():
        filter_shape = filters.get_shape().as_list()
      else:
        filter_shape = array_ops.shape(filters)
      filter_height, filter_width = filter_shape[0], filter_shape[1]
      # Spatial dimensions of the filters and the upsampled filters in which we
      # introduce (rate - 1) zeros between consecutive filter values.
      filter_height_up = filter_height + (filter_height - 1) * (rate - 1)
      filter_width_up = filter_width + (filter_width - 1) * (rate - 1)
      pad_height = filter_height_up - 1
      pad_width = filter_width_up - 1
      # When pad_height (pad_width) is odd, we pad more to bottom (right),
      # following the same convention as conv2d().
      pad_top = pad_height // 2
      pad_bottom = pad_height - pad_top
      pad_left = pad_width // 2
      pad_right = pad_width - pad_left
    elif padding == "VALID":
      pad_top = 0
      pad_bottom = 0
      pad_left = 0
      pad_right = 0
    else:
      raise ValueError("`padding` must be either 'VALID' or 'SAME'. "
                       f"Received: padding={padding}")
    in_height = output_shape[1] + pad_top + pad_bottom
    in_width = output_shape[2] + pad_left + pad_right
    # More padding so that rate divides the height and width of the input.
    pad_bottom_extra = (rate - in_height % rate) % rate
    pad_right_extra = (rate - in_width % rate) % rate
    # The paddings argument to space_to_batch is just the extra padding
    # component.
    space_to_batch_pad = [[0, pad_bottom_extra], [0, pad_right_extra]]
    value = array_ops.space_to_batch(
        input=value, paddings=space_to_batch_pad, block_size=rate)
    input_sizes = [
        rate * rate * output_shape[0], (in_height + pad_bottom_extra) // rate,
        (in_width + pad_right_extra) // rate, output_shape[3]
    ]
    value = gen_nn_ops.conv2d_backprop_input(
        input_sizes=input_sizes,
        filter=filters,
        out_backprop=value,
        strides=[1, 1, 1, 1],
        padding="VALID",
        data_format="NHWC")
    # The crops argument to batch_to_space includes both padding components.
    batch_to_space_crop = [[pad_top, pad_bottom + pad_bottom_extra],
                           [pad_left, pad_right + pad_right_extra]]
    return array_ops.batch_to_space(
        input=value, crops=batch_to_space_crop, block_size=rate)
