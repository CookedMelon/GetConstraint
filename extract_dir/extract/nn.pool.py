@tf_export("nn.pool", v1=[])
@dispatch.add_dispatch_support
def pool_v2(
    input,  # pylint: disable=redefined-builtin
    window_shape,
    pooling_type,
    strides=None,
    padding="VALID",
    data_format=None,
    dilations=None,
    name=None):
  # pylint: disable=line-too-long
  """Performs an N-D pooling operation.
  In the case that `data_format` does not start with "NC", computes for
      0 <= b < batch_size,
      0 <= x[i] < output_spatial_shape[i],
      0 <= c < num_channels:
  ```
  output[b, x[0], ..., x[N-1], c] =
    REDUCE_{z[0], ..., z[N-1]}
      input[b,
            x[0] * strides[0] - pad_before[0] + dilation_rate[0]*z[0],
            ...
            x[N-1]*strides[N-1] - pad_before[N-1] + dilation_rate[N-1]*z[N-1],
            c],
  ```
  where the reduction function REDUCE depends on the value of `pooling_type`,
  and pad_before is defined based on the value of `padding` as described in
  the "returns" section of `tf.nn.convolution` for details.
  The reduction never includes out-of-bounds positions.
  In the case that `data_format` starts with `"NC"`, the `input` and output are
  simply transposed as follows:
  ```python
  pool(input, data_format, **kwargs) =
    tf.transpose(pool(tf.transpose(input, [0] + range(2,N+2) + [1]),
                      **kwargs),
                 [0, N+1] + range(1, N+1))
  ```
  Args:
    input: Tensor of rank N+2, of shape `[batch_size] + input_spatial_shape +
      [num_channels]` if data_format does not start with "NC" (default), or
      `[batch_size, num_channels] + input_spatial_shape` if data_format starts
      with "NC".  Pooling happens over the spatial dimensions only.
    window_shape: Sequence of N ints >= 1.
    pooling_type: Specifies pooling operation, must be "AVG" or "MAX".
    strides: Optional. Sequence of N ints >= 1.  Defaults to `[1]*N`. If any value of
      strides is > 1, then all values of dilation_rate must be 1.
    padding: The padding algorithm, must be "SAME" or "VALID". Defaults to "SAME".
      See
      [here](https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2)
      for more information.
    data_format: A string or None.  Specifies whether the channel dimension of
      the `input` and output is the last dimension (default, or if `data_format`
      does not start with "NC"), or the second dimension (if `data_format`
      starts with "NC").  For N=1, the valid values are "NWC" (default) and
      "NCW".  For N=2, the valid values are "NHWC" (default) and "NCHW". For
      N=3, the valid values are "NDHWC" (default) and "NCDHW".
    dilations: Optional.  Dilation rate.  List of N ints >= 1. Defaults to
      `[1]*N`.  If any value of dilation_rate is > 1, then all values of strides
      must be 1.
    name: Optional. Name of the op.
  Returns:
    Tensor of rank N+2, of shape
      [batch_size] + output_spatial_shape + [num_channels]
    if data_format is None or does not start with "NC", or
      [batch_size, num_channels] + output_spatial_shape
    if data_format starts with "NC",
    where `output_spatial_shape` depends on the value of padding:
    If padding = "SAME":
      output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides[i])
    If padding = "VALID":
      output_spatial_shape[i] =
        ceil((input_spatial_shape[i] - (window_shape[i] - 1) * dilation_rate[i])
             / strides[i]).
  Raises:
    ValueError: if arguments are invalid.
  """
  return pool(
      input=input,
      window_shape=window_shape,
      pooling_type=pooling_type,
      padding=padding,
      dilation_rate=dilations,
      strides=strides,
      name=name,
      data_format=data_format)
