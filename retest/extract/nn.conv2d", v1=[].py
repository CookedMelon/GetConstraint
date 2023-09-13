@tf_export("nn.conv2d", v1=[])
@dispatch.add_dispatch_support
def conv2d_v2(input,  # pylint: disable=redefined-builtin
              filters,
              strides,
              padding,
              data_format="NHWC",
              dilations=None,
              name=None):
  # pylint: disable=line-too-long
  r"""Computes a 2-D convolution given `input` and 4-D `filters` tensors.
  The `input` tensor may have rank `4` or higher, where shape dimensions `[:-3]`
  are considered batch dimensions (`batch_shape`).
  Given an input tensor of shape
  `batch_shape + [in_height, in_width, in_channels]` and a filter / kernel
  tensor of shape `[filter_height, filter_width, in_channels, out_channels]`,
  this op performs the following:
  1. Flattens the filter to a 2-D matrix with shape
     `[filter_height * filter_width * in_channels, output_channels]`.
  2. Extracts image patches from the input tensor to form a *virtual*
     tensor of shape `[batch, out_height, out_width,
     filter_height * filter_width * in_channels]`.
  3. For each patch, right-multiplies the filter matrix and the image patch
     vector.
  In detail, with the default NHWC format,
      output[b, i, j, k] =
          sum_{di, dj, q} input[b, strides[1] * i + di, strides[2] * j + dj, q] *
                          filter[di, dj, q, k]
  Must have `strides[0] = strides[3] = 1`.  For the most common case of the same
  horizontal and vertical strides, `strides = [1, stride, stride, 1]`.
  Usage Example:
  >>> x_in = np.array([[
  ...   [[2], [1], [2], [0], [1]],
  ...   [[1], [3], [2], [2], [3]],
  ...   [[1], [1], [3], [3], [0]],
  ...   [[2], [2], [0], [1], [1]],
  ...   [[0], [0], [3], [1], [2]], ]])
  >>> kernel_in = np.array([
  ...  [ [[2, 0.1]], [[3, 0.2]] ],
  ...  [ [[0, 0.3]], [[1, 0.4]] ], ])
  >>> x = tf.constant(x_in, dtype=tf.float32)
  >>> kernel = tf.constant(kernel_in, dtype=tf.float32)
  >>> tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding='VALID')
  <tf.Tensor: shape=(1, 4, 4, 2), dtype=float32, numpy=..., dtype=float32)>
  Args:
    input: A `Tensor`. Must be one of the following types:
      `half`, `bfloat16`, `float32`, `float64`.
      A Tensor of rank at least 4. The dimension order is interpreted according
      to the value of `data_format`; with the all-but-inner-3 dimensions acting
      as batch dimensions. See below for details.
    filters: A `Tensor`. Must have the same type as `input`.
      A 4-D tensor of shape
      `[filter_height, filter_width, in_channels, out_channels]`
    strides: An int or list of `ints` that has length `1`, `2` or `4`.  The
      stride of the sliding window for each dimension of `input`. If a single
      value is given it is replicated in the `H` and `W` dimension. By default
      the `N` and `C` dimensions are set to 1. The dimension order is determined
      by the value of `data_format`, see below for details.
    padding: Either the `string` `"SAME"` or `"VALID"` indicating the type of
      padding algorithm to use, or a list indicating the explicit paddings at
      the start and end of each dimension. See
      [here](https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2)
      for more information. When explicit padding is used and data_format is
      `"NHWC"`, this should be in the form `[[0, 0], [pad_top, pad_bottom],
      [pad_left, pad_right], [0, 0]]`. When explicit padding used and
      data_format is `"NCHW"`, this should be in the form `[[0, 0], [0, 0],
      [pad_top, pad_bottom], [pad_left, pad_right]]`.
    data_format: An optional `string` from: `"NHWC", "NCHW"`.
      Defaults to `"NHWC"`.
      Specify the data format of the input and output data. With the
      default format "NHWC", the data is stored in the order of:
          `batch_shape + [height, width, channels]`.
      Alternatively, the format could be "NCHW", the data storage order of:
          `batch_shape + [channels, height, width]`.
    dilations: An int or list of `ints` that has length `1`, `2` or `4`,
      defaults to 1. The dilation factor for each dimension of`input`. If a
      single value is given it is replicated in the `H` and `W` dimension. By
      default the `N` and `C` dimensions are set to 1. If set to k > 1, there
      will be k-1 skipped cells between each filter element on that dimension.
      The dimension order is determined by the value of `data_format`, see above
      for details. Dilations in the batch and depth dimensions if a 4-d tensor
      must be 1.
    name: A name for the operation (optional).
  Returns:
    A `Tensor`. Has the same type as `input` and the same outer batch shape.
  """
  # pylint: enable=line-too-long
  return conv2d(input,  # pylint: disable=redefined-builtin
                filters,
                strides,
                padding,
                use_cudnn_on_gpu=True,
                data_format=data_format,
                dilations=dilations,
                name=name)
