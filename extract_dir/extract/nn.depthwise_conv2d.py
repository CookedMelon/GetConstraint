@tf_export("nn.depthwise_conv2d", v1=[])
@dispatch.add_dispatch_support
def depthwise_conv2d_v2(input,
                        filter,
                        strides,
                        padding,
                        data_format=None,
                        dilations=None,
                        name=None):
  """Depthwise 2-D convolution.
  Given a 4D input tensor ('NHWC' or 'NCHW' data formats)
  and a filter tensor of shape
  `[filter_height, filter_width, in_channels, channel_multiplier]`
  containing `in_channels` convolutional filters of depth 1, `depthwise_conv2d`
  applies a different filter to each input channel (expanding from 1 channel
  to `channel_multiplier` channels for each), then concatenates the results
  together.  The output has `in_channels * channel_multiplier` channels.
  In detail, with the default NHWC format,
      output[b, i, j, k * channel_multiplier + q] =
          sum_{di, dj} filter[di, dj, k, q] *
                       input[b, strides[1] * i + dilations[0] * di,
                                strides[2] * j + dilations[1] * dj, k]
  Must have `strides[0] = strides[3] = 1`.  For the most common case of the
  same horizontal and vertical strides, `strides = [1, stride, stride, 1]`.
  If any value in `dilations` is greater than 1, we perform atrous depthwise
  convolution, in which case all values in the `strides` tensor must be equal
  to 1.
  Usage Example:
  >>> x = np.array([
  ...     [1., 2.],
  ...     [3., 4.],
  ...     [5., 6.]
  ... ], dtype=np.float32).reshape((1, 3, 2, 1))
  >>> kernel = np.array([
  ...     [1., 2.],
  ...     [3., 4]
  ... ], dtype=np.float32).reshape((2, 1, 1, 2))
  >>> tf.nn.depthwise_conv2d(x, kernel, strides=[1, 1, 1, 1],
  ...                        padding='VALID').numpy()
    array([[[[10., 14.],
             [14., 20.]],
            [[18., 26.],
             [22., 32.]]]], dtype=float32)
  >>> tf.nn.depthwise_conv2d(x, kernel, strides=[1, 1, 1, 1],
  ...                        padding=[[0, 0], [1, 0], [1, 0], [0, 0]]).numpy()
    array([[[[ 0.,  0.],
             [ 3.,  4.],
             [ 6.,  8.]],
            [[ 0.,  0.],
             [10., 14.],
             [14., 20.]],
            [[ 0.,  0.],
             [18., 26.],
             [22., 32.]]]], dtype=float32)
  Args:
    input: 4-D with shape according to `data_format`.
    filter: 4-D with shape
      `[filter_height, filter_width, in_channels, channel_multiplier]`.
    strides: 1-D of size 4.  The stride of the sliding window for each
      dimension of `input`.
    padding: Controls how to pad the image before applying the convolution. Can
      be the string `"SAME"` or `"VALID"` indicating the type of padding
      algorithm to use, or a list indicating the explicit paddings at the start
      and end of each dimension. See
      [here](https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2)
      for more information. When explicit padding is used and data_format
      is `"NHWC"`, this should be in the form `[[0, 0], [pad_top, pad_bottom],
      [pad_left, pad_right], [0, 0]]`. When explicit padding used and
      data_format is `"NCHW"`, this should be in the form `[[0, 0], [0, 0],
      [pad_top, pad_bottom], [pad_left, pad_right]]`.
    data_format: The data format for input. Either "NHWC" (default) or "NCHW".
    dilations: 1-D of size 2. The dilation rate in which we sample input values
      across the `height` and `width` dimensions in atrous convolution. If it is
      greater than 1, then all values of strides must be 1.
    name: A name for this operation (optional).
  Returns:
    A 4-D `Tensor` with shape according to `data_format`.  E.g., for
    "NHWC" format, shape is
    `[batch, out_height, out_width, in_channels * channel_multiplier].`
  """
  return depthwise_conv2d(input=input,
                          filter=filter,
                          strides=strides,
                          padding=padding,
                          rate=dilations,
                          name=name,
                          data_format=data_format)
