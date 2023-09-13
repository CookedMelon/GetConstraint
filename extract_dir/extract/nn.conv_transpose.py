@tf_export("nn.conv_transpose")
@dispatch.add_dispatch_support
def conv_transpose(input,  # pylint: disable=redefined-builtin
                   filters,
                   output_shape,
                   strides,
                   padding="SAME",
                   data_format=None,
                   dilations=None,
                   name=None):
  """The transpose of `convolution`.
  This operation is sometimes called "deconvolution" after
  (Zeiler et al., 2010), but is really the transpose (gradient) of `conv3d`
  rather than an actual deconvolution.
  Args:
    input: An N+2 dimensional `Tensor` of shape
      `[batch_size] + input_spatial_shape + [in_channels]` if data_format does
      not start with "NC" (default), or
      `[batch_size, in_channels] + input_spatial_shape` if data_format starts
      with "NC". It must be one of the following types:
      `half`, `bfloat16`, `float32`, `float64`.
    filters: An N+2 dimensional `Tensor` with the same type as `input` and
      shape `spatial_filter_shape + [in_channels, out_channels]`.
    output_shape: A 1-D `Tensor` representing the output shape of the
      deconvolution op.
    strides: An int or list of `ints` that has length `1`, `N` or `N+2`.  The
      stride of the sliding window for each dimension of `input`. If a single
      value is given it is replicated in the spatial dimensions. By default
      the `N` and `C` dimensions are set to 0. The dimension order is determined
      by the value of `data_format`, see below for details.
    padding: A string, either `'VALID'` or `'SAME'`. The padding algorithm. See
      [here](https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2)
      for more information.
    data_format: A string or None.  Specifies whether the channel dimension of
      the `input` and output is the last dimension (default, or if `data_format`
      does not start with "NC"), or the second dimension (if `data_format`
      starts with "NC").  For N=1, the valid values are "NWC" (default) and
      "NCW".  For N=2, the valid values are "NHWC" (default) and "NCHW".
      For N=3, the valid values are "NDHWC" (default) and "NCDHW".
    dilations: An int or list of `ints` that has length `1`, `N` or `N+2`,
      defaults to 1. The dilation factor for each dimension of`input`. If a
      single value is given it is replicated in the spatial dimensions. By
      default the `N` and `C` dimensions are set to 1. If set to k > 1, there
      will be k-1 skipped cells between each filter element on that dimension.
      The dimension order is determined by the value of `data_format`, see above
      for details.
    name: A name for the operation (optional). If not specified "conv_transpose"
      is used.
  Returns:
    A `Tensor` with the same type as `value`.
  References:
    Deconvolutional Networks:
      [Zeiler et al., 2010]
      (https://ieeexplore.ieee.org/abstract/document/5539957)
      ([pdf]
      (http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.232.4023&rep=rep1&type=pdf))
  """
  with ops.name_scope(name, "conv_transpose",
                      [input, filter, output_shape]) as name:
    if tensor_util.is_tf_type(output_shape):
      n = output_shape.shape[0] - 2
    elif isinstance(output_shape, collections_abc.Sized):
      n = len(output_shape) - 2
    else:
      raise ValueError("`output_shape` must be a tensor or sized collection. "
                       f"Received: output_shape={output_shape}")
    if not 1 <= n <= 3:
      raise ValueError(
          f"`output_shape` must be of length 3, 4 or 5. "
          f"Received: output_shape={output_shape} of length {n + 2}.")
    op = CONV_TRANSPOSE_OPS[n-1]
    return op(
        input,
        filters,
        output_shape,
        strides,
        padding=padding,
        data_format=data_format,
        dilations=dilations,
        name=name)
