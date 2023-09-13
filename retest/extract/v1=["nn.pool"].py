@tf_export(v1=["nn.pool"])
@dispatch.add_dispatch_support
def pool(
    input,  # pylint: disable=redefined-builtin
    window_shape,
    pooling_type,
    padding,
    dilation_rate=None,
    strides=None,
    name=None,
    data_format=None,
    dilations=None):
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
    input: Tensor of rank N+2, of shape
      `[batch_size] + input_spatial_shape + [num_channels]` if data_format does
      not start with "NC" (default), or
      `[batch_size, num_channels] + input_spatial_shape` if data_format starts
      with "NC".  Pooling happens over the spatial dimensions only.
    window_shape: Sequence of N ints >= 1.
    pooling_type: Specifies pooling operation, must be "AVG" or "MAX".
    padding: The padding algorithm, must be "SAME" or "VALID".
      See the "returns" section of `tf.nn.convolution` for details.
    dilation_rate: Optional.  Dilation rate.  List of N ints >= 1.
      Defaults to `[1]*N`.  If any value of dilation_rate is > 1, then all
      values of strides must be 1.
    strides: Optional.  Sequence of N ints >= 1.  Defaults to `[1]*N`.
      If any value of strides is > 1, then all values of dilation_rate must be
      1.
    name: Optional. Name of the op.
    data_format: A string or None.  Specifies whether the channel dimension of
      the `input` and output is the last dimension (default, or if `data_format`
      does not start with "NC"), or the second dimension (if `data_format`
      starts with "NC").  For N=1, the valid values are "NWC" (default) and
      "NCW".  For N=2, the valid values are "NHWC" (default) and "NCHW".
      For N=3, the valid values are "NDHWC" (default) and "NCDHW".
    dilations: Alias for dilation_rate
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
  dilation_rate = deprecated_argument_lookup(
      "dilations", dilations, "dilation_rate", dilation_rate)
  # pylint: enable=line-too-long
  with ops.name_scope(name, "%s_pool" % (pooling_type.lower()),
                      [input]) as scope:
    input = ops.convert_to_tensor(input, name="input")  # pylint: disable=redefined-builtin
    num_spatial_dims = len(window_shape)
    if num_spatial_dims < 1 or num_spatial_dims > 3:
      raise ValueError("`len(window_shape)` must be 1, 2, or 3. Received: "
                       f"window_shape={window_shape} of length "
                       f"{len(window_shape)}")
    input.get_shape().with_rank(num_spatial_dims + 2)
    strides, dilation_rate = _get_strides_and_dilation_rate(
        num_spatial_dims, strides, dilation_rate)
    if padding == "SAME" and np.any(dilation_rate > 1):
      raise ValueError(
          "pooling with 'SAME' padding is not implemented for "
          f"`dilation_rate` > 1. Received: padding={padding} and "
          f"dilation_rate={dilation_rate}")
    if np.any(strides > window_shape):
      raise ValueError(
          "`strides` > `window_shape` not supported due to inconsistency "
          f"between CPU and GPU implementations. Received: strides={strides} "
          f"and window_shape={window_shape}")
    pooling_ops = {
        ("MAX", 1): max_pool,
        ("MAX", 2): max_pool,
        ("MAX", 3): max_pool3d,  # pylint: disable=undefined-variable
        ("AVG", 1): avg_pool,
        ("AVG", 2): avg_pool,
        ("AVG", 3): avg_pool3d,  # pylint: disable=undefined-variable
    }
    op_key = (pooling_type, num_spatial_dims)
    if op_key not in pooling_ops:
      raise ValueError(
          f"{num_spatial_dims}-D {pooling_type} pooling is not supported.")
    if data_format is None or not data_format.startswith("NC"):
      adjusted_window_shape = [1] + list(window_shape) + [1]
      adjusted_strides = [1] + list(strides) + [1]
      spatial_dims = range(1, num_spatial_dims + 1)
    else:
      adjusted_window_shape = [1, 1] + list(window_shape)
      adjusted_strides = [1, 1] + list(strides)
      spatial_dims = range(2, num_spatial_dims + 2)
    if num_spatial_dims == 1:
      if data_format is None or data_format == "NWC":
        data_format_kwargs = dict(data_format="NHWC")
      elif data_format == "NCW":
        data_format_kwargs = dict(data_format="NCHW")
      else:
        raise ValueError("data_format must be either 'NWC' or 'NCW'. "
                         f"Received: data_format={data_format}")
      adjusted_window_shape = [1] + adjusted_window_shape
      adjusted_strides = [1] + adjusted_strides
    else:
      data_format_kwargs = dict(data_format=data_format)
    def op(converted_input, _, converted_padding):  # pylint: disable=missing-docstring
      if num_spatial_dims == 1:
        converted_input = array_ops.expand_dims(converted_input,
                                                spatial_dims[0])
      result = pooling_ops[op_key](
          converted_input,
          adjusted_window_shape,
          adjusted_strides,
          converted_padding,
          name=scope,
          **data_format_kwargs)
      if num_spatial_dims == 1:
        result = array_ops.squeeze(result, [spatial_dims[0]])
      return result
    return with_space_to_batch(
        input=input,
        dilation_rate=dilation_rate,
        padding=padding,
        op=op,
        spatial_dims=spatial_dims,
        filter_shape=window_shape)
