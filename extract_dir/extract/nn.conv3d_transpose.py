@tf_export("nn.conv3d_transpose", v1=[])
@dispatch.add_dispatch_support
def conv3d_transpose_v2(input,  # pylint: disable=redefined-builtin
                        filters,
                        output_shape,
                        strides,
                        padding="SAME",
                        data_format="NDHWC",
                        dilations=None,
                        name=None):
  """The transpose of `conv3d`.
  This operation is sometimes called "deconvolution" after
  (Zeiler et al., 2010), but is really the transpose (gradient) of `conv3d`
  rather than an actual deconvolution.
  Args:
    input: A 5-D `Tensor` of type `float` and shape `[batch, depth, height,
      width, in_channels]` for `NDHWC` data format or `[batch, in_channels,
      depth, height, width]` for `NCDHW` data format.
    filters: A 5-D `Tensor` with the same type as `input` and shape `[depth,
      height, width, output_channels, in_channels]`.  `filter`'s `in_channels`
      dimension must match that of `input`.
    output_shape: A 1-D `Tensor` representing the output shape of the
      deconvolution op.
    strides: An int or list of `ints` that has length `1`, `3` or `5`.  The
      stride of the sliding window for each dimension of `input`. If a single
      value is given it is replicated in the `D`, `H` and `W` dimension. By
      default the `N` and `C` dimensions are set to 0. The dimension order is
      determined by the value of `data_format`, see below for details.
    padding: A string, either `'VALID'` or `'SAME'`. The padding algorithm. See
      [here](https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2)
      for more information.
    data_format: A string. 'NDHWC' and 'NCDHW' are supported.
    dilations: An int or list of `ints` that has length `1`, `3` or `5`,
      defaults to 1. The dilation factor for each dimension of`input`. If a
      single value is given it is replicated in the `D`, `H` and `W` dimension.
      By default the `N` and `C` dimensions are set to 1. If set to k > 1, there
      will be k-1 skipped cells between each filter element on that dimension.
      The dimension order is determined by the value of `data_format`, see above
      for details. Dilations in the batch and depth dimensions if a 5-d tensor
      must be 1.
    name: Optional name for the returned tensor.
  Returns:
    A `Tensor` with the same type as `input`.
  References:
    Deconvolutional Networks:
      [Zeiler et al., 2010]
      (https://ieeexplore.ieee.org/abstract/document/5539957)
      ([pdf]
      (http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.232.4023&rep=rep1&type=pdf))
  """
  with ops.name_scope(name, "conv3d_transpose",
                      [input, filter, output_shape]) as name:
    if data_format is None:
      data_format = "NDHWC"
    channel_index = 1 if data_format.startswith("NC") else 4
    strides = _get_sequence(strides, 3, channel_index, "strides")
    dilations = _get_sequence(dilations, 3, channel_index, "dilations")
    return gen_nn_ops.conv3d_backprop_input_v2(
        input_sizes=output_shape,
        filter=filters,
        out_backprop=input,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilations=dilations,
        name=name)
