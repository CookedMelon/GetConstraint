@tf_export("nn.max_pool", v1=["nn.max_pool_v2"])
@dispatch.add_dispatch_support
def max_pool_v2(input, ksize, strides, padding, data_format=None, name=None):
  """Performs max pooling on the input.
  For a given window of `ksize`, takes the maximum value within that window.
  Used for reducing computation and preventing overfitting.
  Consider an example of pooling with 2x2, non-overlapping windows:
  >>> matrix = tf.constant([
  ...     [0, 0, 1, 7],
  ...     [0, 2, 0, 0],
  ...     [5, 2, 0, 0],
  ...     [0, 0, 9, 8],
  ... ])
  >>> reshaped = tf.reshape(matrix, (1, 4, 4, 1))
  >>> tf.nn.max_pool(reshaped, ksize=2, strides=2, padding="SAME")
  <tf.Tensor: shape=(1, 2, 2, 1), dtype=int32, numpy=
  array([[[[2],
           [7]],
          [[5],
           [9]]]], dtype=int32)>
  We can adjust the window size using the `ksize` parameter. For example, if we
  were to expand the window to 3:
  >>> tf.nn.max_pool(reshaped, ksize=3, strides=2, padding="SAME")
  <tf.Tensor: shape=(1, 2, 2, 1), dtype=int32, numpy=
  array([[[[5],
           [7]],
          [[9],
           [9]]]], dtype=int32)>
  We've now picked up two additional large numbers (5 and 9) in two of the
  pooled spots.
  Note that our windows are now overlapping, since we're still moving by 2 units
  on each iteration. This is causing us to see the same 9 repeated twice, since
  it is part of two overlapping windows.
  We can adjust how far we move our window with each iteration using the
  `strides` parameter. Updating this to the same value as our window size
  eliminates the overlap:
  >>> tf.nn.max_pool(reshaped, ksize=3, strides=3, padding="SAME")
  <tf.Tensor: shape=(1, 2, 2, 1), dtype=int32, numpy=
  array([[[[2],
           [7]],
          [[5],
           [9]]]], dtype=int32)>
  Because the window does not neatly fit into our input, padding is added around
  the edges, giving us the same result as when we used a 2x2 window. We can skip
  padding altogether and simply drop the windows that do not fully fit into our
  input by instead passing `"VALID"` to the `padding` argument:
  >>> tf.nn.max_pool(reshaped, ksize=3, strides=3, padding="VALID")
  <tf.Tensor: shape=(1, 1, 1, 1), dtype=int32, numpy=array([[[[5]]]],
   dtype=int32)>
  Now we've grabbed the largest value in the 3x3 window starting from the upper-
  left corner. Since no other windows fit in our input, they are dropped.
  Args:
    input:  Tensor of rank N+2, of shape `[batch_size] + input_spatial_shape +
      [num_channels]` if `data_format` does not start with "NC" (default), or
      `[batch_size, num_channels] + input_spatial_shape` if data_format starts
      with "NC". Pooling happens over the spatial dimensions only.
    ksize: An int or list of `ints` that has length `1`, `N` or `N+2`. The size
      of the window for each dimension of the input tensor.
    strides: An int or list of `ints` that has length `1`, `N` or `N+2`. The
      stride of the sliding window for each dimension of the input tensor.
    padding: Either the `string` `"SAME"` or `"VALID"` indicating the type of
      padding algorithm to use, or a list indicating the explicit paddings at
      the start and end of each dimension. See
      [here](https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2)
      for more information. When explicit padding is used and data_format is
      `"NHWC"`, this should be in the form `[[0, 0], [pad_top, pad_bottom],
      [pad_left, pad_right], [0, 0]]`. When explicit padding used and
      data_format is `"NCHW"`, this should be in the form `[[0, 0], [0, 0],
      [pad_top, pad_bottom], [pad_left, pad_right]]`. When using explicit
      padding, the size of the paddings cannot be greater than the sliding
      window size.
    data_format: A string. Specifies the channel dimension. For N=1 it can be
      either "NWC" (default) or "NCW", for N=2 it can be either "NHWC" (default)
      or "NCHW" and for N=3 either "NDHWC" (default) or "NCDHW".
    name: Optional name for the operation.
  Returns:
    A `Tensor` of format specified by `data_format`.
    The max pooled output tensor.
  Raises:
    ValueError: If
      - explicit padding is used with an input tensor of rank 5.
      - explicit padding is used with data_format='NCHW_VECT_C'.
  """
  if input.shape is not None:
    n = len(input.shape) - 2
  elif data_format is not None:
    n = len(data_format) - 2
  else:
    raise ValueError(
        "`input` must have a static shape or a data format must be given. "
        f"Received: input.shape={input.shape} and "
        f"data_format={data_format}")
  if not 1 <= n <= 3:
    raise ValueError(
        f"`input.shape.rank` must be 3, 4 or 5. Received: "
        f"input.shape={input.shape} of rank {n + 2}.")
  if data_format is None:
    channel_index = n + 1
  else:
    channel_index = 1 if data_format.startswith("NC") else n + 1
  if isinstance(padding, (list, tuple)) and data_format == "NCHW_VECT_C":
    raise ValueError("`data_format='NCHW_VECT_C'` is not supported with "
                     f"explicit padding. Received: padding={padding}")
  ksize = _get_sequence(ksize, n, channel_index, "ksize")
  strides = _get_sequence(strides, n, channel_index, "strides")
  if (isinstance(padding, (list, tuple)) and n == 3):
    raise ValueError("Explicit padding is not supported with an input "
                     f"tensor of rank 5. Received: padding={padding}")
  max_pooling_ops = {
      1: max_pool1d,
      2: max_pool2d,
      3: gen_nn_ops.max_pool3d
  }
  op = max_pooling_ops[n]
  return op(
      input,
      ksize=ksize,
      strides=strides,
      padding=padding,
      data_format=data_format,
      name=name)
