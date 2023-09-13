@tf_export("nn.max_pool2d")
@dispatch.add_dispatch_support
def max_pool2d(input, ksize, strides, padding, data_format="NHWC", name=None):
  """Performs max pooling on 2D spatial data such as images.
  This is a more specific version of `tf.nn.max_pool` where the input tensor
  is 4D, representing 2D spatial data such as images. Using these APIs are
  equivalent
  Downsamples the input images along theirs spatial dimensions (height and
  width) by taking its maximum over an input window defined by `ksize`.
  The window is shifted by `strides` along each dimension.
  For example, for `strides=(2, 2)` and `padding=VALID` windows that extend
  outside of the input are not included in the output:
  >>> x = tf.constant([[1., 2., 3., 4.],
  ...                  [5., 6., 7., 8.],
  ...                  [9., 10., 11., 12.]])
  >>> # Add the `batch` and `channels` dimensions.
  >>> x = x[tf.newaxis, :, :, tf.newaxis]
  >>> result = tf.nn.max_pool2d(x, ksize=(2, 2), strides=(2, 2),
  ...                           padding="VALID")
  >>> result[0, :, :, 0]
  <tf.Tensor: shape=(1, 2), dtype=float32, numpy=
  array([[6., 8.]], dtype=float32)>
  With `padding=SAME`, we get:
  >>> x = tf.constant([[1., 2., 3., 4.],
  ...                  [5., 6., 7., 8.],
  ...                  [9., 10., 11., 12.]])
  >>> x = x[tf.newaxis, :, :, tf.newaxis]
  >>> result = tf.nn.max_pool2d(x, ksize=(2, 2), strides=(2, 2),
  ...                           padding='SAME')
  >>> result[0, :, :, 0]
  <tf.Tensor: shape=(2, 2), dtype=float32, numpy=
  array([[ 6., 8.],
         [10.,12.]], dtype=float32)>
  We can also specify padding explicitly. The following example adds width-1
  padding on all sides (top, bottom, left, right):
  >>> x = tf.constant([[1., 2., 3., 4.],
  ...                  [5., 6., 7., 8.],
  ...                  [9., 10., 11., 12.]])
  >>> x = x[tf.newaxis, :, :, tf.newaxis]
  >>> result = tf.nn.max_pool2d(x, ksize=(2, 2), strides=(2, 2),
  ...                           padding=[[0, 0], [1, 1], [1, 1], [0, 0]])
  >>> result[0, :, :, 0]
  <tf.Tensor: shape=(2, 3), dtype=float32, numpy=
  array([[ 1., 3., 4.],
         [ 9., 11., 12.]], dtype=float32)>
  For more examples and detail, see `tf.nn.max_pool`.
  Args:
    input: A 4-D `Tensor` of the format specified by `data_format`.
    ksize: An int or list of `ints` that has length `1`, `2` or `4`. The size of
      the window for each dimension of the input tensor. If only one integer is
      specified, then we apply the same window for all 4 dims. If two are
      provided then we use those for H, W dimensions and keep N, C dimension
      window size = 1.
    strides: An int or list of `ints` that has length `1`, `2` or `4`. The
      stride of the sliding window for each dimension of the input tensor. If
      only one integer is specified, we apply the same stride to all 4 dims. If
      two are provided we use those for the H, W dimensions and keep N, C of
      stride = 1.
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
    data_format: A string. 'NHWC', 'NCHW' and 'NCHW_VECT_C' are supported.
    name: Optional name for the operation.
  Returns:
    A `Tensor` of format specified by `data_format`.
    The max pooled output tensor.
  Raises:
    ValueError: If explicit padding is used with data_format='NCHW_VECT_C'.
  """
  with ops.name_scope(name, "MaxPool2d", [input]) as name:
    if data_format is None:
      data_format = "NHWC"
    channel_index = 1 if data_format.startswith("NC") else 3
    ksize = _get_sequence(ksize, 2, channel_index, "ksize")
    strides = _get_sequence(strides, 2, channel_index, "strides")
    if isinstance(padding, (list, tuple)) and data_format == "NCHW_VECT_C":
      raise ValueError("`data_format='NCHW_VECT_C'` is not supported with "
                       f"explicit padding. Received: padding={padding}")
    padding, explicit_paddings = convert_padding(padding)
    return gen_nn_ops.max_pool(
        input,
        ksize=ksize,
        strides=strides,
        padding=padding,
        explicit_paddings=explicit_paddings,
        data_format=data_format,
        name=name)
