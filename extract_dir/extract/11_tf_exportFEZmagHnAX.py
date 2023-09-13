"/home/cc/Workspace/tfconstraint/python/ops/image_ops_impl.py"
@tf_export(
    'image.resize_with_crop_or_pad',
    v1=['image.resize_with_crop_or_pad', 'image.resize_image_with_crop_or_pad'])
@dispatch.add_dispatch_support
def resize_image_with_crop_or_pad(image, target_height, target_width):
  """Crops and/or pads an image to a target width and height.
  Resizes an image to a target width and height by either centrally
  cropping the image or padding it evenly with zeros.
  If `width` or `height` is greater than the specified `target_width` or
  `target_height` respectively, this op centrally crops along that dimension.
  For example:
  >>> image = np.arange(75).reshape(5, 5, 3)  # create 3-D image input
  >>> image[:,:,0]  # print first channel just for demo purposes
  array([[ 0,  3,  6,  9, 12],
         [15, 18, 21, 24, 27],
         [30, 33, 36, 39, 42],
         [45, 48, 51, 54, 57],
         [60, 63, 66, 69, 72]])
  >>> image = tf.image.resize_with_crop_or_pad(image, 3, 3)  # crop
  >>> # print first channel for demo purposes; centrally cropped output
  >>> image[:,:,0]
  <tf.Tensor: shape=(3, 3), dtype=int64, numpy=
  array([[18, 21, 24],
         [33, 36, 39],
         [48, 51, 54]])>
  If `width` or `height` is smaller than the specified `target_width` or
  `target_height` respectively, this op centrally pads with 0 along that
  dimension.
  For example:
  >>> image = np.arange(1, 28).reshape(3, 3, 3)  # create 3-D image input
  >>> image[:,:,0]  # print first channel just for demo purposes
  array([[ 1,  4,  7],
         [10, 13, 16],
         [19, 22, 25]])
  >>> image = tf.image.resize_with_crop_or_pad(image, 5, 5)  # pad
  >>> # print first channel for demo purposes; we should see 0 paddings
  >>> image[:,:,0]
  <tf.Tensor: shape=(5, 5), dtype=int64, numpy=
  array([[ 0,  0,  0,  0,  0],
         [ 0,  1,  4,  7,  0],
         [ 0, 10, 13, 16,  0],
         [ 0, 19, 22, 25,  0],
         [ 0,  0,  0,  0,  0]])>
  Args:
    image: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor
      of shape `[height, width, channels]`.
    target_height: Target height.
    target_width: Target width.
  Raises:
    ValueError: if `target_height` or `target_width` are zero or negative.
  Returns:
    Cropped and/or padded image.
    If `images` was 4-D, a 4-D float Tensor of shape
    `[batch, new_height, new_width, channels]`.
    If `images` was 3-D, a 3-D float Tensor of shape
    `[new_height, new_width, channels]`.
  """
  with ops.name_scope(None, 'resize_image_with_crop_or_pad', [image]):
    image = ops.convert_to_tensor(image, name='image')
    image_shape = image.get_shape()
    is_batch = True
    if image_shape.ndims == 3:
      is_batch = False
      image = array_ops.expand_dims(image, 0)
    elif image_shape.ndims is None:
      is_batch = False
      image = array_ops.expand_dims(image, 0)
      image.set_shape([None] * 4)
    elif image_shape.ndims != 4:
      raise ValueError(
          '\'image\' (shape %s) must have either 3 or 4 dimensions.' %
          image_shape)
    assert_ops = _CheckAtLeast3DImage(image, require_static=False)
    assert_ops += _assert(target_width > 0, ValueError,
                          'target_width must be > 0.')
    assert_ops += _assert(target_height > 0, ValueError,
                          'target_height must be > 0.')
    image = control_flow_ops.with_dependencies(assert_ops, image)
    # `crop_to_bounding_box` and `pad_to_bounding_box` have their own checks.
    # Make sure our checks come first, so that error messages are clearer.
    if _is_tensor(target_height):
      target_height = control_flow_ops.with_dependencies(
          assert_ops, target_height)
    if _is_tensor(target_width):
      target_width = control_flow_ops.with_dependencies(assert_ops,
                                                        target_width)
    def max_(x, y):
      if _is_tensor(x) or _is_tensor(y):
        return math_ops.maximum(x, y)
      else:
        return max(x, y)
    def min_(x, y):
      if _is_tensor(x) or _is_tensor(y):
        return math_ops.minimum(x, y)
      else:
        return min(x, y)
    def equal_(x, y):
      if _is_tensor(x) or _is_tensor(y):
        return math_ops.equal(x, y)
      else:
        return x == y
    _, height, width, _ = _ImageDimensions(image, rank=4)
    width_diff = target_width - width
    offset_crop_width = max_(-width_diff // 2, 0)
    offset_pad_width = max_(width_diff // 2, 0)
    height_diff = target_height - height
    offset_crop_height = max_(-height_diff // 2, 0)
    offset_pad_height = max_(height_diff // 2, 0)
    # Maybe crop if needed.
    cropped = crop_to_bounding_box(image, offset_crop_height, offset_crop_width,
                                   min_(target_height, height),
                                   min_(target_width, width))
    # Maybe pad if needed.
    resized = pad_to_bounding_box(cropped, offset_pad_height, offset_pad_width,
                                  target_height, target_width)
    # In theory all the checks below are redundant.
    if resized.get_shape().ndims is None:
      raise ValueError('resized contains no shape.')
    _, resized_height, resized_width, _ = _ImageDimensions(resized, rank=4)
    assert_ops = []
    assert_ops += _assert(
        equal_(resized_height, target_height), ValueError,
        'resized height is not correct.')
    assert_ops += _assert(
        equal_(resized_width, target_width), ValueError,
        'resized width is not correct.')
    resized = control_flow_ops.with_dependencies(assert_ops, resized)
    if not is_batch:
      resized = array_ops.squeeze(resized, axis=[0])
    return resized
@tf_export(v1=['image.ResizeMethod'])
class ResizeMethodV1:
  """See `v1.image.resize` for details."""
  BILINEAR = 0
  NEAREST_NEIGHBOR = 1
  BICUBIC = 2
  AREA = 3
@tf_export('image.ResizeMethod', v1=[])
class ResizeMethod:
  """See `tf.image.resize` for details."""
  BILINEAR = 'bilinear'
  NEAREST_NEIGHBOR = 'nearest'
  BICUBIC = 'bicubic'
  AREA = 'area'
  LANCZOS3 = 'lanczos3'
  LANCZOS5 = 'lanczos5'
  GAUSSIAN = 'gaussian'
  MITCHELLCUBIC = 'mitchellcubic'
def _resize_images_common(images, resizer_fn, size, preserve_aspect_ratio, name,
                          skip_resize_if_same):
  """Core functionality for v1 and v2 resize functions."""
  with ops.name_scope(name, 'resize', [images, size]):
    images = ops.convert_to_tensor(images, name='images')
    if images.get_shape().ndims is None:
      raise ValueError('\'images\' contains no shape.')
    # TODO(shlens): Migrate this functionality to the underlying Op's.
    is_batch = True
    if images.get_shape().ndims == 3:
      is_batch = False
      images = array_ops.expand_dims(images, 0)
    elif images.get_shape().ndims != 4:
      raise ValueError('\'images\' must have either 3 or 4 dimensions.')
    _, height, width, _ = images.get_shape().as_list()
    try:
      size = ops.convert_to_tensor(size, dtypes.int32, name='size')
    except (TypeError, ValueError):
      raise ValueError('\'size\' must be a 1-D int32 Tensor')
    if not size.get_shape().is_compatible_with([2]):
      raise ValueError('\'size\' must be a 1-D Tensor of 2 elements: '
                       'new_height, new_width')
    if preserve_aspect_ratio:
      # Get the current shapes of the image, even if dynamic.
      _, current_height, current_width, _ = _ImageDimensions(images, rank=4)
      # do the computation to find the right scale and height/width.
      scale_factor_height = (
          math_ops.cast(size[0], dtypes.float32) /
          math_ops.cast(current_height, dtypes.float32))
      scale_factor_width = (
          math_ops.cast(size[1], dtypes.float32) /
          math_ops.cast(current_width, dtypes.float32))
      scale_factor = math_ops.minimum(scale_factor_height, scale_factor_width)
      scaled_height_const = math_ops.cast(
          math_ops.round(scale_factor *
                         math_ops.cast(current_height, dtypes.float32)),
          dtypes.int32)
      scaled_width_const = math_ops.cast(
          math_ops.round(scale_factor *
                         math_ops.cast(current_width, dtypes.float32)),
          dtypes.int32)
      # NOTE: Reset the size and other constants used later.
      size = ops.convert_to_tensor([scaled_height_const, scaled_width_const],
                                   dtypes.int32,
                                   name='size')
    size_const_as_shape = tensor_util.constant_value_as_shape(size)
    new_height_const = tensor_shape.dimension_at_index(size_const_as_shape,
                                                       0).value
    new_width_const = tensor_shape.dimension_at_index(size_const_as_shape,
                                                      1).value
    # If we can determine that the height and width will be unmodified by this
    # transformation, we avoid performing the resize.
    if skip_resize_if_same and all(
        x is not None
        for x in [new_width_const, width, new_height_const, height]) and (
            width == new_width_const and height == new_height_const):
      if not is_batch:
        images = array_ops.squeeze(images, axis=[0])
      return images
    images = resizer_fn(images, size)
    # NOTE(mrry): The shape functions for the resize ops cannot unpack
    # the packed values in `new_size`, so set the shape here.
    images.set_shape([None, new_height_const, new_width_const, None])
    if not is_batch:
      images = array_ops.squeeze(images, axis=[0])
    return images
@tf_export(v1=['image.resize_images', 'image.resize'])
@dispatch.add_dispatch_support
def resize_images(images,
                  size,
                  method=ResizeMethodV1.BILINEAR,
                  align_corners=False,
                  preserve_aspect_ratio=False,
                  name=None):
  """Resize `images` to `size` using the specified `method`.
  Resized images will be distorted if their original aspect ratio is not
  the same as `size`.  To avoid distortions see
  `tf.image.resize_with_pad` or `tf.image.resize_with_crop_or_pad`.
  The `method` can be one of:
  *   <b>`tf.image.ResizeMethod.BILINEAR`</b>: [Bilinear interpolation.](
    https://en.wikipedia.org/wiki/Bilinear_interpolation)
  *   <b>`tf.image.ResizeMethod.NEAREST_NEIGHBOR`</b>: [
    Nearest neighbor interpolation.](
    https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation)
  *   <b>`tf.image.ResizeMethod.BICUBIC`</b>: [Bicubic interpolation.](
    https://en.wikipedia.org/wiki/Bicubic_interpolation)
  *   <b>`tf.image.ResizeMethod.AREA`</b>: Area interpolation.
  The return value has the same type as `images` if `method` is
  `tf.image.ResizeMethod.NEAREST_NEIGHBOR`. It will also have the same type
  as `images` if the size of `images` can be statically determined to be the
  same as `size`, because `images` is returned in this case. Otherwise, the
  return value has type `float32`.
  Args:
    images: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor
      of shape `[height, width, channels]`.
    size: A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The new
      size for the images.
    method: ResizeMethod.  Defaults to `tf.image.ResizeMethod.BILINEAR`.
    align_corners: bool.  If True, the centers of the 4 corner pixels of the
      input and output tensors are aligned, preserving the values at the corner
      pixels. Defaults to `False`.
    preserve_aspect_ratio: Whether to preserve the aspect ratio. If this is set,
      then `images` will be resized to a size that fits in `size` while
      preserving the aspect ratio of the original image. Scales up the image if
      `size` is bigger than the current size of the `image`. Defaults to False.
    name: A name for this operation (optional).
  Raises:
    ValueError: if the shape of `images` is incompatible with the
      shape arguments to this function
    ValueError: if `size` has invalid shape or type.
    ValueError: if an unsupported resize method is specified.
  Returns:
    If `images` was 4-D, a 4-D float Tensor of shape
    `[batch, new_height, new_width, channels]`.
    If `images` was 3-D, a 3-D float Tensor of shape
    `[new_height, new_width, channels]`.
  """
  def resize_fn(images_t, new_size):
    """Legacy resize core function, passed to _resize_images_common."""
    if method == ResizeMethodV1.BILINEAR or method == ResizeMethod.BILINEAR:
      return gen_image_ops.resize_bilinear(
          images_t, new_size, align_corners=align_corners)
    elif (method == ResizeMethodV1.NEAREST_NEIGHBOR or
          method == ResizeMethod.NEAREST_NEIGHBOR):
      return gen_image_ops.resize_nearest_neighbor(
          images_t, new_size, align_corners=align_corners)
    elif method == ResizeMethodV1.BICUBIC or method == ResizeMethod.BICUBIC:
      return gen_image_ops.resize_bicubic(
          images_t, new_size, align_corners=align_corners)
    elif method == ResizeMethodV1.AREA or method == ResizeMethod.AREA:
      return gen_image_ops.resize_area(
          images_t, new_size, align_corners=align_corners)
    else:
      raise ValueError('Resize method is not implemented: {}'.format(method))
  return _resize_images_common(
      images,
      resize_fn,
      size,
      preserve_aspect_ratio=preserve_aspect_ratio,
      name=name,
      skip_resize_if_same=True)
@tf_export('image.resize', v1=[])
@dispatch.add_dispatch_support
def resize_images_v2(images,
                     size,
                     method=ResizeMethod.BILINEAR,
                     preserve_aspect_ratio=False,
                     antialias=False,
                     name=None):
  """Resize `images` to `size` using the specified `method`.
  Resized images will be distorted if their original aspect ratio is not
  the same as `size`.  To avoid distortions see
  `tf.image.resize_with_pad`.
  >>> image = tf.constant([
  ...  [1,0,0,0,0],
  ...  [0,1,0,0,0],
  ...  [0,0,1,0,0],
  ...  [0,0,0,1,0],
  ...  [0,0,0,0,1],
  ... ])
  >>> # Add "batch" and "channels" dimensions
  >>> image = image[tf.newaxis, ..., tf.newaxis]
  >>> image.shape.as_list()  # [batch, height, width, channels]
  [1, 5, 5, 1]
  >>> tf.image.resize(image, [3,5])[0,...,0].numpy()
  array([[0.6666667, 0.3333333, 0.       , 0.       , 0.       ],
         [0.       , 0.       , 1.       , 0.       , 0.       ],
         [0.       , 0.       , 0.       , 0.3333335, 0.6666665]],
        dtype=float32)
  It works equally well with a single image instead of a batch of images:
  >>> tf.image.resize(image[0], [3,5]).shape.as_list()
  [3, 5, 1]
  When `antialias` is true, the sampling filter will anti-alias the input image
  as well as interpolate.  When downsampling an image with [anti-aliasing](
  https://en.wikipedia.org/wiki/Spatial_anti-aliasing) the sampling filter
  kernel is scaled in order to properly anti-alias the input image signal.
  `antialias` has no effect when upsampling an image:
  >>> a = tf.image.resize(image, [5,10])
  >>> b = tf.image.resize(image, [5,10], antialias=True)
  >>> tf.reduce_max(abs(a - b)).numpy()
  0.0
  The `method` argument expects an item from the `image.ResizeMethod` enum, or
  the string equivalent. The options are:
  *   <b>`bilinear`</b>: [Bilinear interpolation.](
    https://en.wikipedia.org/wiki/Bilinear_interpolation) If `antialias` is
    true, becomes a hat/tent filter function with radius 1 when downsampling.
  *   <b>`lanczos3`</b>:  [Lanczos kernel](
    https://en.wikipedia.org/wiki/Lanczos_resampling) with radius 3.
    High-quality practical filter but may have some ringing, especially on
    synthetic images.
  *   <b>`lanczos5`</b>: [Lanczos kernel] (
    https://en.wikipedia.org/wiki/Lanczos_resampling) with radius 5.
    Very-high-quality filter but may have stronger ringing.
  *   <b>`bicubic`</b>: [Cubic interpolant](
    https://en.wikipedia.org/wiki/Bicubic_interpolation) of Keys. Equivalent to
    Catmull-Rom kernel. Reasonably good quality and faster than Lanczos3Kernel,
    particularly when upsampling.
  *   <b>`gaussian`</b>: [Gaussian kernel](
    https://en.wikipedia.org/wiki/Gaussian_filter) with radius 3,
    sigma = 1.5 / 3.0.
  *   <b>`nearest`</b>: [Nearest neighbor interpolation.](
    https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation)
    `antialias` has no effect when used with nearest neighbor interpolation.
  *   <b>`area`</b>: Anti-aliased resampling with area interpolation.
    `antialias` has no effect when used with area interpolation; it
    always anti-aliases.
  *   <b>`mitchellcubic`</b>: Mitchell-Netravali Cubic non-interpolating filter.
    For synthetic images (especially those lacking proper prefiltering), less
    ringing than Keys cubic kernel but less sharp.
  Note: Near image edges the filtering kernel may be partially outside the
  image boundaries. For these pixels, only input pixels inside the image will be
  included in the filter sum, and the output value will be appropriately
  normalized.
  The return value has type `float32`, unless the `method` is
  `ResizeMethod.NEAREST_NEIGHBOR`, then the return dtype is the dtype
  of `images`:
  >>> nn = tf.image.resize(image, [5,7], method='nearest')
  >>> nn[0,...,0].numpy()
  array([[1, 0, 0, 0, 0, 0, 0],
         [0, 1, 1, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 1, 1, 0],
         [0, 0, 0, 0, 0, 0, 1]], dtype=int32)
  With `preserve_aspect_ratio=True`, the aspect ratio is preserved, so `size`
  is the maximum for each dimension:
  >>> max_10_20 = tf.image.resize(image, [10,20], preserve_aspect_ratio=True)
  >>> max_10_20.shape.as_list()
  [1, 10, 10, 1]
  Args:
    images: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor
      of shape `[height, width, channels]`.
    size: A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The new
      size for the images.
    method: An `image.ResizeMethod`, or string equivalent.  Defaults to
      `bilinear`.
    preserve_aspect_ratio: Whether to preserve the aspect ratio. If this is set,
      then `images` will be resized to a size that fits in `size` while
      preserving the aspect ratio of the original image. Scales up the image if
      `size` is bigger than the current size of the `image`. Defaults to False.
    antialias: Whether to use an anti-aliasing filter when downsampling an
      image.
    name: A name for this operation (optional).
  Raises:
    ValueError: if the shape of `images` is incompatible with the
      shape arguments to this function
    ValueError: if `size` has an invalid shape or type.
    ValueError: if an unsupported resize method is specified.
  Returns:
    If `images` was 4-D, a 4-D float Tensor of shape
    `[batch, new_height, new_width, channels]`.
    If `images` was 3-D, a 3-D float Tensor of shape
    `[new_height, new_width, channels]`.
  """
  def resize_fn(images_t, new_size):
    """Resize core function, passed to _resize_images_common."""
    scale_and_translate_methods = [
        ResizeMethod.LANCZOS3, ResizeMethod.LANCZOS5, ResizeMethod.GAUSSIAN,
        ResizeMethod.MITCHELLCUBIC
    ]
    def resize_with_scale_and_translate(method):
      scale = (
          math_ops.cast(new_size, dtype=dtypes.float32) /
          math_ops.cast(array_ops.shape(images_t)[1:3], dtype=dtypes.float32))
      return gen_image_ops.scale_and_translate(
          images_t,
          new_size,
          scale,
          array_ops.zeros([2]),
          kernel_type=method,
          antialias=antialias)
    if method == ResizeMethod.BILINEAR:
      if antialias:
        return resize_with_scale_and_translate('triangle')
      else:
        return gen_image_ops.resize_bilinear(
            images_t, new_size, half_pixel_centers=True)
    elif method == ResizeMethod.NEAREST_NEIGHBOR:
      return gen_image_ops.resize_nearest_neighbor(
          images_t, new_size, half_pixel_centers=True)
    elif method == ResizeMethod.BICUBIC:
      if antialias:
        return resize_with_scale_and_translate('keyscubic')
      else:
        return gen_image_ops.resize_bicubic(
            images_t, new_size, half_pixel_centers=True)
    elif method == ResizeMethod.AREA:
      return gen_image_ops.resize_area(images_t, new_size)
    elif method in scale_and_translate_methods:
      return resize_with_scale_and_translate(method)
    else:
      raise ValueError('Resize method is not implemented: {}'.format(method))
  return _resize_images_common(
      images,
      resize_fn,
      size,
      preserve_aspect_ratio=preserve_aspect_ratio,
      name=name,
      skip_resize_if_same=False)
def _resize_image_with_pad_common(image, target_height, target_width,
                                  resize_fn):
  """Core functionality for v1 and v2 resize_image_with_pad functions."""
  with ops.name_scope(None, 'resize_image_with_pad', [image]):
    image = ops.convert_to_tensor(image, name='image')
    image_shape = image.get_shape()
    is_batch = True
    if image_shape.ndims == 3:
      is_batch = False
      image = array_ops.expand_dims(image, 0)
    elif image_shape.ndims is None:
      is_batch = False
      image = array_ops.expand_dims(image, 0)
      image.set_shape([None] * 4)
    elif image_shape.ndims != 4:
      raise ValueError(
          '\'image\' (shape %s) must have either 3 or 4 dimensions.' %
          image_shape)
    assert_ops = _CheckAtLeast3DImage(image, require_static=False)
    assert_ops += _assert(target_width > 0, ValueError,
                          'target_width must be > 0.')
    assert_ops += _assert(target_height > 0, ValueError,
                          'target_height must be > 0.')
    image = control_flow_ops.with_dependencies(assert_ops, image)
    def max_(x, y):
      if _is_tensor(x) or _is_tensor(y):
        return math_ops.maximum(x, y)
      else:
        return max(x, y)
    _, height, width, _ = _ImageDimensions(image, rank=4)
    # convert values to float, to ease divisions
    f_height = math_ops.cast(height, dtype=dtypes.float32)
    f_width = math_ops.cast(width, dtype=dtypes.float32)
    f_target_height = math_ops.cast(target_height, dtype=dtypes.float32)
    f_target_width = math_ops.cast(target_width, dtype=dtypes.float32)
    # Find the ratio by which the image must be adjusted
    # to fit within the target
    ratio = max_(f_width / f_target_width, f_height / f_target_height)
    resized_height_float = f_height / ratio
    resized_width_float = f_width / ratio
    resized_height = math_ops.cast(
        math_ops.floor(resized_height_float), dtype=dtypes.int32)
    resized_width = math_ops.cast(
        math_ops.floor(resized_width_float), dtype=dtypes.int32)
    padding_height = (f_target_height - resized_height_float) / 2
    padding_width = (f_target_width - resized_width_float) / 2
    f_padding_height = math_ops.floor(padding_height)
    f_padding_width = math_ops.floor(padding_width)
    p_height = max_(0, math_ops.cast(f_padding_height, dtype=dtypes.int32))
    p_width = max_(0, math_ops.cast(f_padding_width, dtype=dtypes.int32))
    # Resize first, then pad to meet requested dimensions
    resized = resize_fn(image, [resized_height, resized_width])
    padded = pad_to_bounding_box(resized, p_height, p_width, target_height,
                                 target_width)
    if padded.get_shape().ndims is None:
      raise ValueError('padded contains no shape.')
    _ImageDimensions(padded, rank=4)
    if not is_batch:
      padded = array_ops.squeeze(padded, axis=[0])
    return padded
@tf_export(v1=['image.resize_image_with_pad'])
@dispatch.add_dispatch_support
def resize_image_with_pad_v1(image,
                             target_height,
                             target_width,
                             method=ResizeMethodV1.BILINEAR,
                             align_corners=False):
  """Resizes and pads an image to a target width and height.
  Resizes an image to a target width and height by keeping
  the aspect ratio the same without distortion. If the target
  dimensions don't match the image dimensions, the image
  is resized and then padded with zeroes to match requested
  dimensions.
  Args:
    image: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor
      of shape `[height, width, channels]`.
    target_height: Target height.
    target_width: Target width.
    method: Method to use for resizing image. See `resize_images()`
    align_corners: bool.  If True, the centers of the 4 corner pixels of the
      input and output tensors are aligned, preserving the values at the corner
      pixels. Defaults to `False`.
  Raises:
    ValueError: if `target_height` or `target_width` are zero or negative.
  Returns:
    Resized and padded image.
    If `images` was 4-D, a 4-D float Tensor of shape
    `[batch, new_height, new_width, channels]`.
    If `images` was 3-D, a 3-D float Tensor of shape
    `[new_height, new_width, channels]`.
  """
  def _resize_fn(im, new_size):
    return resize_images(im, new_size, method, align_corners=align_corners)
  return _resize_image_with_pad_common(image, target_height, target_width,
                                       _resize_fn)
@tf_export('image.resize_with_pad', v1=[])
@dispatch.add_dispatch_support
def resize_image_with_pad_v2(image,
                             target_height,
                             target_width,
                             method=ResizeMethod.BILINEAR,
                             antialias=False):
  """Resizes and pads an image to a target width and height.
  Resizes an image to a target width and height by keeping
  the aspect ratio the same without distortion. If the target
  dimensions don't match the image dimensions, the image
  is resized and then padded with zeroes to match requested
  dimensions.
  Args:
    image: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor
      of shape `[height, width, channels]`.
    target_height: Target height.
    target_width: Target width.
    method: Method to use for resizing image. See `image.resize()`
    antialias: Whether to use anti-aliasing when resizing. See 'image.resize()'.
  Raises:
    ValueError: if `target_height` or `target_width` are zero or negative.
  Returns:
    Resized and padded image.
    If `images` was 4-D, a 4-D float Tensor of shape
    `[batch, new_height, new_width, channels]`.
    If `images` was 3-D, a 3-D float Tensor of shape
    `[new_height, new_width, channels]`.
  """
  def _resize_fn(im, new_size):
    return resize_images_v2(im, new_size, method, antialias=antialias)
  return _resize_image_with_pad_common(image, target_height, target_width,
                                       _resize_fn)
@tf_export('image.per_image_standardization')
@dispatch.add_dispatch_support
def per_image_standardization(image):
  """Linearly scales each image in `image` to have mean 0 and variance 1.
  For each 3-D image `x` in `image`, computes `(x - mean) / adjusted_stddev`,
  where
  - `mean` is the average of all values in `x`
  - `adjusted_stddev = max(stddev, 1.0/sqrt(N))` is capped away from 0 to
    protect against division by 0 when handling uniform images
    - `N` is the number of elements in `x`
    - `stddev` is the standard deviation of all values in `x`
  Example Usage:
  >>> image = tf.constant(np.arange(1, 13, dtype=np.int32), shape=[2, 2, 3])
  >>> image # 3-D tensor
  <tf.Tensor: shape=(2, 2, 3), dtype=int32, numpy=
  array([[[ 1,  2,  3],
          [ 4,  5,  6]],
         [[ 7,  8,  9],
          [10, 11, 12]]], dtype=int32)>
  >>> new_image = tf.image.per_image_standardization(image)
  >>> new_image # 3-D tensor with mean ~= 0 and variance ~= 1
  <tf.Tensor: shape=(2, 2, 3), dtype=float32, numpy=
  array([[[-1.593255  , -1.3035723 , -1.0138896 ],
          [-0.7242068 , -0.4345241 , -0.14484136]],
         [[ 0.14484136,  0.4345241 ,  0.7242068 ],
          [ 1.0138896 ,  1.3035723 ,  1.593255  ]]], dtype=float32)>
  Args:
    image: An n-D `Tensor` with at least 3 dimensions, the last 3 of which are
      the dimensions of each image.
  Returns:
    A `Tensor` with the same shape as `image` and its dtype is `float32`.
  Raises:
    ValueError: The shape of `image` has fewer than 3 dimensions.
  """
  with ops.name_scope(None, 'per_image_standardization', [image]) as scope:
    image = ops.convert_to_tensor(image, name='image')
    image = _AssertAtLeast3DImage(image)
    image = math_ops.cast(image, dtype=dtypes.float32)
    num_pixels = math_ops.reduce_prod(array_ops.shape(image)[-3:])
    image_mean = math_ops.reduce_mean(image, axis=[-1, -2, -3], keepdims=True)
    # Apply a minimum normalization that protects us against uniform images.
    stddev = math_ops.reduce_std(image, axis=[-1, -2, -3], keepdims=True)
    min_stddev = math_ops.rsqrt(math_ops.cast(num_pixels, dtypes.float32))
    adjusted_stddev = math_ops.maximum(stddev, min_stddev)
    image -= image_mean
    image = math_ops.divide(image, adjusted_stddev, name=scope)
    return image
@tf_export('image.random_brightness')
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def random_brightness(image, max_delta, seed=None):
  """Adjust the brightness of images by a random factor.
  Equivalent to `adjust_brightness()` using a `delta` randomly picked in the
  interval `[-max_delta, max_delta)`.
  For producing deterministic results given a `seed` value, use
  `tf.image.stateless_random_brightness`. Unlike using the `seed` param
  with `tf.image.random_*` ops, `tf.image.stateless_random_*` ops guarantee the
  same results given the same seed independent of how many times the function is
  called, and independent of global seed settings (e.g. tf.random.set_seed).
  Args:
    image: An image or images to adjust.
    max_delta: float, must be non-negative.
    seed: A Python integer. Used to create a random seed. See
      `tf.compat.v1.set_random_seed` for behavior.
  Usage Example:
  >>> x = [[[1.0, 2.0, 3.0],
  ...       [4.0, 5.0, 6.0]],
  ...      [[7.0, 8.0, 9.0],
  ...       [10.0, 11.0, 12.0]]]
  >>> tf.image.random_brightness(x, 0.2)
  <tf.Tensor: shape=(2, 2, 3), dtype=float32, numpy=...>
  Returns:
    The brightness-adjusted image(s).
  Raises:
    ValueError: if `max_delta` is negative.
  """
  if max_delta < 0:
    raise ValueError('max_delta must be non-negative.')
  delta = random_ops.random_uniform([], -max_delta, max_delta, seed=seed)
  return adjust_brightness(image, delta)
@tf_export('image.stateless_random_brightness', v1=[])
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def stateless_random_brightness(image, max_delta, seed):
  """Adjust the brightness of images by a random factor deterministically.
  Equivalent to `adjust_brightness()` using a `delta` randomly picked in the
  interval `[-max_delta, max_delta)`.
  Guarantees the same results given the same `seed` independent of how many
  times the function is called, and independent of global seed settings (e.g.
  `tf.random.set_seed`).
  Usage Example:
  >>> x = [[[1.0, 2.0, 3.0],
  ...       [4.0, 5.0, 6.0]],
  ...      [[7.0, 8.0, 9.0],
  ...       [10.0, 11.0, 12.0]]]
  >>> seed = (1, 2)
  >>> tf.image.stateless_random_brightness(x, 0.2, seed)
  <tf.Tensor: shape=(2, 2, 3), dtype=float32, numpy=
  array([[[ 1.1376241,  2.1376243,  3.1376243],
          [ 4.1376243,  5.1376243,  6.1376243]],
         [[ 7.1376243,  8.137624 ,  9.137624 ],
          [10.137624 , 11.137624 , 12.137624 ]]], dtype=float32)>
  Args:
    image: An image or images to adjust.
    max_delta: float, must be non-negative.
    seed: A shape [2] Tensor, the seed to the random number generator. Must have
      dtype `int32` or `int64`. (When using XLA, only `int32` is allowed.)
  Returns:
    The brightness-adjusted image(s).
  Raises:
    ValueError: if `max_delta` is negative.
  """
  if max_delta < 0:
    raise ValueError('max_delta must be non-negative.')
  delta = stateless_random_ops.stateless_random_uniform(
      shape=[], minval=-max_delta, maxval=max_delta, seed=seed)
  return adjust_brightness(image, delta)
@tf_export('image.random_contrast')
@dispatch.add_dispatch_support
def random_contrast(image, lower, upper, seed=None):
  """Adjust the contrast of an image or images by a random factor.
  Equivalent to `adjust_contrast()` but uses a `contrast_factor` randomly
  picked in the interval `[lower, upper)`.
  For producing deterministic results given a `seed` value, use
  `tf.image.stateless_random_contrast`. Unlike using the `seed` param
  with `tf.image.random_*` ops, `tf.image.stateless_random_*` ops guarantee the
  same results given the same seed independent of how many times the function is
  called, and independent of global seed settings (e.g. tf.random.set_seed).
  Args:
    image: An image tensor with 3 or more dimensions.
    lower: float.  Lower bound for the random contrast factor.
    upper: float.  Upper bound for the random contrast factor.
    seed: A Python integer. Used to create a random seed. See
      `tf.compat.v1.set_random_seed` for behavior.
  Usage Example:
  >>> x = [[[1.0, 2.0, 3.0],
  ...       [4.0, 5.0, 6.0]],
  ...     [[7.0, 8.0, 9.0],
  ...       [10.0, 11.0, 12.0]]]
  >>> tf.image.random_contrast(x, 0.2, 0.5)
  <tf.Tensor: shape=(2, 2, 3), dtype=float32, numpy=...>
  Returns:
    The contrast-adjusted image(s).
  Raises:
    ValueError: if `upper <= lower` or if `lower < 0`.
  """
  if upper <= lower:
    raise ValueError('upper must be > lower.')
  if lower < 0:
    raise ValueError('lower must be non-negative.')
  contrast_factor = random_ops.random_uniform([], lower, upper, seed=seed)
  return adjust_contrast(image, contrast_factor)
@tf_export('image.stateless_random_contrast', v1=[])
@dispatch.add_dispatch_support
def stateless_random_contrast(image, lower, upper, seed):
  """Adjust the contrast of images by a random factor deterministically.
  Guarantees the same results given the same `seed` independent of how many
  times the function is called, and independent of global seed settings (e.g.
  `tf.random.set_seed`).
  Args:
    image: An image tensor with 3 or more dimensions.
    lower: float.  Lower bound for the random contrast factor.
    upper: float.  Upper bound for the random contrast factor.
    seed: A shape [2] Tensor, the seed to the random number generator. Must have
      dtype `int32` or `int64`. (When using XLA, only `int32` is allowed.)
  Usage Example:
  >>> x = [[[1.0, 2.0, 3.0],
  ...       [4.0, 5.0, 6.0]],
  ...      [[7.0, 8.0, 9.0],
  ...       [10.0, 11.0, 12.0]]]
  >>> seed = (1, 2)
  >>> tf.image.stateless_random_contrast(x, 0.2, 0.5, seed)
  <tf.Tensor: shape=(2, 2, 3), dtype=float32, numpy=
  array([[[3.4605184, 4.4605184, 5.4605184],
          [4.820173 , 5.820173 , 6.820173 ]],
         [[6.179827 , 7.179827 , 8.179828 ],
          [7.5394816, 8.539482 , 9.539482 ]]], dtype=float32)>
  Returns:
    The contrast-adjusted image(s).
  Raises:
    ValueError: if `upper <= lower` or if `lower < 0`.
  """
  if upper <= lower:
    raise ValueError('upper must be > lower.')
  if lower < 0:
    raise ValueError('lower must be non-negative.')
  contrast_factor = stateless_random_ops.stateless_random_uniform(
      shape=[], minval=lower, maxval=upper, seed=seed)
  return adjust_contrast(image, contrast_factor)
@tf_export('image.adjust_brightness')
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def adjust_brightness(image, delta):
  """Adjust the brightness of RGB or Grayscale images.
  This is a convenience method that converts RGB images to float
  representation, adjusts their brightness, and then converts them back to the
  original data type. If several adjustments are chained, it is advisable to
  minimize the number of redundant conversions.
  The value `delta` is added to all components of the tensor `image`. `image` is
  converted to `float` and scaled appropriately if it is in fixed-point
  representation, and `delta` is converted to the same data type. For regular
  images, `delta` should be in the range `(-1,1)`, as it is added to the image
  in floating point representation, where pixel values are in the `[0,1)` range.
  Usage Example:
  >>> x = [[[1.0, 2.0, 3.0],
  ...       [4.0, 5.0, 6.0]],
  ...     [[7.0, 8.0, 9.0],
  ...       [10.0, 11.0, 12.0]]]
  >>> tf.image.adjust_brightness(x, delta=0.1)
  <tf.Tensor: shape=(2, 2, 3), dtype=float32, numpy=
  array([[[ 1.1,  2.1,  3.1],
          [ 4.1,  5.1,  6.1]],
         [[ 7.1,  8.1,  9.1],
          [10.1, 11.1, 12.1]]], dtype=float32)>
  Args:
    image: RGB image or images to adjust.
    delta: A scalar. Amount to add to the pixel values.
  Returns:
    A brightness-adjusted tensor of the same shape and type as `image`.
  """
  with ops.name_scope(None, 'adjust_brightness', [image, delta]) as name:
    image = ops.convert_to_tensor(image, name='image')
    # Remember original dtype to so we can convert back if needed
    orig_dtype = image.dtype
    if orig_dtype in [dtypes.float16, dtypes.float32]:
      flt_image = image
    else:
      flt_image = convert_image_dtype(image, dtypes.float32)
    adjusted = math_ops.add(
        flt_image, math_ops.cast(delta, flt_image.dtype), name=name)
    return convert_image_dtype(adjusted, orig_dtype, saturate=True)
@tf_export('image.adjust_contrast')
@dispatch.add_dispatch_support
def adjust_contrast(images, contrast_factor):
  """Adjust contrast of RGB or grayscale images.
  This is a convenience method that converts RGB images to float
  representation, adjusts their contrast, and then converts them back to the
  original data type. If several adjustments are chained, it is advisable to
  minimize the number of redundant conversions.
  `images` is a tensor of at least 3 dimensions.  The last 3 dimensions are
  interpreted as `[height, width, channels]`.  The other dimensions only
  represent a collection of images, such as `[batch, height, width, channels].`
  Contrast is adjusted independently for each channel of each image.
  For each channel, this Op computes the mean of the image pixels in the
  channel and then adjusts each component `x` of each pixel to
  `(x - mean) * contrast_factor + mean`.
  `contrast_factor` must be in the interval `(-inf, inf)`.
  Usage Example:
  >>> x = [[[1.0, 2.0, 3.0],
  ...       [4.0, 5.0, 6.0]],
  ...     [[7.0, 8.0, 9.0],
  ...       [10.0, 11.0, 12.0]]]
  >>> tf.image.adjust_contrast(x, 2.)
  <tf.Tensor: shape=(2, 2, 3), dtype=float32, numpy=
  array([[[-3.5, -2.5, -1.5],
          [ 2.5,  3.5,  4.5]],
         [[ 8.5,  9.5, 10.5],
          [14.5, 15.5, 16.5]]], dtype=float32)>
  Args:
    images: Images to adjust.  At least 3-D.
    contrast_factor: A float multiplier for adjusting contrast.
  Returns:
    The contrast-adjusted image or images.
  """
  with ops.name_scope(None, 'adjust_contrast',
                      [images, contrast_factor]) as name:
    images = ops.convert_to_tensor(images, name='images')
    # Remember original dtype to so we can convert back if needed
    orig_dtype = images.dtype
    if orig_dtype in (dtypes.float16, dtypes.float32):
      flt_images = images
    else:
      flt_images = convert_image_dtype(images, dtypes.float32)
    adjusted = gen_image_ops.adjust_contrastv2(
        flt_images, contrast_factor=contrast_factor, name=name)
    return convert_image_dtype(adjusted, orig_dtype, saturate=True)
@tf_export('image.adjust_gamma')
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def adjust_gamma(image, gamma=1, gain=1):
  """Performs [Gamma Correction](http://en.wikipedia.org/wiki/Gamma_correction).
  on the input image.
  Also known as Power Law Transform. This function converts the
  input images at first to float representation, then transforms them
  pixelwise according to the equation `Out = gain * In**gamma`,
  and then converts the back to the original data type.
  Usage Example:
  >>> x = [[[1.0, 2.0, 3.0],
  ...       [4.0, 5.0, 6.0]],
  ...     [[7.0, 8.0, 9.0],
  ...       [10.0, 11.0, 12.0]]]
  >>> tf.image.adjust_gamma(x, 0.2)
  <tf.Tensor: shape=(2, 2, 3), dtype=float32, numpy=
  array([[[1.       , 1.1486983, 1.2457309],
          [1.319508 , 1.3797297, 1.4309691]],
         [[1.4757731, 1.5157166, 1.5518456],
          [1.5848932, 1.6153942, 1.6437519]]], dtype=float32)>
  Args:
    image : RGB image or images to adjust.
    gamma : A scalar or tensor. Non-negative real number.
    gain  : A scalar or tensor. The constant multiplier.
  Returns:
    A Tensor. A Gamma-adjusted tensor of the same shape and type as `image`.
  Raises:
    ValueError: If gamma is negative.
  Notes:
    For gamma greater than 1, the histogram will shift towards left and
    the output image will be darker than the input image.
    For gamma less than 1, the histogram will shift towards right and
    the output image will be brighter than the input image.
  References:
    [Wikipedia](http://en.wikipedia.org/wiki/Gamma_correction)
  """
  with ops.name_scope(None, 'adjust_gamma', [image, gamma, gain]) as name:
    image = ops.convert_to_tensor(image, name='image')
    # Remember original dtype to so we can convert back if needed
    orig_dtype = image.dtype
    if orig_dtype in [dtypes.float16, dtypes.float32]:
      flt_image = image
    else:
      flt_image = convert_image_dtype(image, dtypes.float32)
    assert_op = _assert(gamma >= 0, ValueError,
                        'Gamma should be a non-negative real number.')
    if assert_op:
      gamma = control_flow_ops.with_dependencies(assert_op, gamma)
    # According to the definition of gamma correction.
    adjusted_img = gain * flt_image**gamma
    return convert_image_dtype(adjusted_img, orig_dtype, saturate=True)
@tf_export('image.convert_image_dtype')
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def convert_image_dtype(image, dtype, saturate=False, name=None):
  """Convert `image` to `dtype`, scaling its values if needed.
  The operation supports data types (for `image` and `dtype`) of
  `uint8`, `uint16`, `uint32`, `uint64`, `int8`, `int16`, `int32`, `int64`,
  `float16`, `float32`, `float64`, `bfloat16`.
  Images that are represented using floating point values are expected to have
  values in the range [0,1). Image data stored in integer data types are
  expected to have values in the range `[0,MAX]`, where `MAX` is the largest
  positive representable number for the data type.
  This op converts between data types, scaling the values appropriately before
  casting.
  Usage Example:
  >>> x = [[[1, 2, 3], [4, 5, 6]],
  ...      [[7, 8, 9], [10, 11, 12]]]
  >>> x_int8 = tf.convert_to_tensor(x, dtype=tf.int8)
  >>> tf.image.convert_image_dtype(x_int8, dtype=tf.float16, saturate=False)
  <tf.Tensor: shape=(2, 2, 3), dtype=float16, numpy=
  array([[[0.00787, 0.01575, 0.02362],
          [0.0315 , 0.03937, 0.04724]],
         [[0.0551 , 0.063  , 0.07086],
          [0.07874, 0.0866 , 0.0945 ]]], dtype=float16)>
  Converting integer types to floating point types returns normalized floating
  point values in the range [0, 1); the values are normalized by the `MAX` value
  of the input dtype. Consider the following two examples:
  >>> a = [[[1], [2]], [[3], [4]]]
  >>> a_int8 = tf.convert_to_tensor(a, dtype=tf.int8)
  >>> tf.image.convert_image_dtype(a_int8, dtype=tf.float32)
  <tf.Tensor: shape=(2, 2, 1), dtype=float32, numpy=
  array([[[0.00787402],
          [0.01574803]],
         [[0.02362205],
          [0.03149606]]], dtype=float32)>
  >>> a_int32 = tf.convert_to_tensor(a, dtype=tf.int32)
  >>> tf.image.convert_image_dtype(a_int32, dtype=tf.float32)
  <tf.Tensor: shape=(2, 2, 1), dtype=float32, numpy=
  array([[[4.6566129e-10],
          [9.3132257e-10]],
         [[1.3969839e-09],
          [1.8626451e-09]]], dtype=float32)>
  Despite having identical values of `a` and output dtype of `float32`, the
  outputs differ due to the different input dtypes (`int8` vs. `int32`). This
  is, again, because the values are normalized by the `MAX` value of the input
  dtype.
  Note that converting floating point values to integer type may lose precision.
  In the example below, an image tensor `b` of dtype `float32` is converted to
  `int8` and back to `float32`. The final output, however, is different from
  the original input `b` due to precision loss.
  >>> b = [[[0.12], [0.34]], [[0.56], [0.78]]]
  >>> b_float32 = tf.convert_to_tensor(b, dtype=tf.float32)
  >>> b_int8 = tf.image.convert_image_dtype(b_float32, dtype=tf.int8)
  >>> tf.image.convert_image_dtype(b_int8, dtype=tf.float32)
  <tf.Tensor: shape=(2, 2, 1), dtype=float32, numpy=
  array([[[0.11811024],
          [0.33858266]],
         [[0.5590551 ],
          [0.77952754]]], dtype=float32)>
  Scaling up from an integer type (input dtype) to another integer type (output
  dtype) will not map input dtype's `MAX` to output dtype's `MAX` but converting
  back and forth should result in no change. For example, as shown below, the
  `MAX` value of int8 (=127) is not mapped to the `MAX` value of int16 (=32,767)
  but, when scaled back, we get the same, original values of `c`.
  >>> c = [[[1], [2]], [[127], [127]]]
  >>> c_int8 = tf.convert_to_tensor(c, dtype=tf.int8)
  >>> c_int16 = tf.image.convert_image_dtype(c_int8, dtype=tf.int16)
  >>> print(c_int16)
  tf.Tensor(
  [[[  256]
    [  512]]
   [[32512]
    [32512]]], shape=(2, 2, 1), dtype=int16)
  >>> c_int8_back = tf.image.convert_image_dtype(c_int16, dtype=tf.int8)
  >>> print(c_int8_back)
  tf.Tensor(
  [[[  1]
    [  2]]
   [[127]
    [127]]], shape=(2, 2, 1), dtype=int8)
  Scaling down from an integer type to another integer type can be a lossy
  conversion. Notice in the example below that converting `int16` to `uint8` and
  back to `int16` has lost precision.
  >>> d = [[[1000], [2000]], [[3000], [4000]]]
  >>> d_int16 = tf.convert_to_tensor(d, dtype=tf.int16)
  >>> d_uint8 = tf.image.convert_image_dtype(d_int16, dtype=tf.uint8)
  >>> d_int16_back = tf.image.convert_image_dtype(d_uint8, dtype=tf.int16)
  >>> print(d_int16_back)
  tf.Tensor(
  [[[ 896]
    [1920]]
   [[2944]
    [3968]]], shape=(2, 2, 1), dtype=int16)
  Note that converting from floating point inputs to integer types may lead to
  over/underflow problems. Set saturate to `True` to avoid such problem in
  problematic conversions. If enabled, saturation will clip the output into the
  allowed range before performing a potentially dangerous cast (and only before
  performing such a cast, i.e., when casting from a floating point to an integer
  type, and when casting from a signed to an unsigned type; `saturate` has no
  effect on casts between floats, or on casts that increase the type's range).
  Args:
    image: An image.
    dtype: A `DType` to convert `image` to.
    saturate: If `True`, clip the input before casting (if necessary).
    name: A name for this operation (optional).
  Returns:
    `image`, converted to `dtype`.
  Raises:
    AttributeError: Raises an attribute error when dtype is neither
    float nor integer.
  """
  image = ops.convert_to_tensor(image, name='image')
  dtype = dtypes.as_dtype(dtype)
  if not dtype.is_floating and not dtype.is_integer:
    raise AttributeError('dtype must be either floating point or integer')
  if not image.dtype.is_floating and not image.dtype.is_integer:
    raise AttributeError('image dtype must be either floating point or integer')
  if dtype == image.dtype:
    return array_ops.identity(image, name=name)
  with ops.name_scope(name, 'convert_image', [image]) as name:
    # Both integer: use integer multiplication in the larger range
    if image.dtype.is_integer and dtype.is_integer:
      scale_in = image.dtype.max
      scale_out = dtype.max
      if scale_in > scale_out:
        # Scaling down, scale first, then cast. The scaling factor will
        # cause in.max to be mapped to above out.max but below out.max+1,
        # so that the output is safely in the supported range.
        scale = (scale_in + 1) // (scale_out + 1)
        scaled = math_ops.floordiv(image, scale)
        if saturate:
          return math_ops.saturate_cast(scaled, dtype, name=name)
        else:
          return math_ops.cast(scaled, dtype, name=name)
      else:
        # Scaling up, cast first, then scale. The scale will not map in.max to
        # out.max, but converting back and forth should result in no change.
        if saturate:
          cast = math_ops.saturate_cast(image, dtype)
        else:
          cast = math_ops.cast(image, dtype)
        scale = (scale_out + 1) // (scale_in + 1)
        return math_ops.multiply(cast, scale, name=name)
    elif image.dtype.is_floating and dtype.is_floating:
      # Both float: Just cast, no possible overflows in the allowed ranges.
      # Note: We're ignoring float overflows. If your image dynamic range
      # exceeds float range, you're on your own.
      return math_ops.cast(image, dtype, name=name)
    else:
      if image.dtype.is_integer:
        # Converting to float: first cast, then scale. No saturation possible.
        cast = math_ops.cast(image, dtype)
        scale = 1. / image.dtype.max
        return math_ops.multiply(cast, scale, name=name)
      else:
        # Converting from float: first scale, then cast
        scale = dtype.max + 0.5  # avoid rounding problems in the cast
        scaled = math_ops.multiply(image, scale)
        if saturate:
          return math_ops.saturate_cast(scaled, dtype, name=name)
        else:
          return math_ops.cast(scaled, dtype, name=name)
@tf_export('image.rgb_to_grayscale')
@dispatch.add_dispatch_support
def rgb_to_grayscale(images, name=None):
  """Converts one or more images from RGB to Grayscale.
  Outputs a tensor of the same `DType` and rank as `images`.  The size of the
  last dimension of the output is 1, containing the Grayscale value of the
  pixels.
  >>> original = tf.constant([[[1.0, 2.0, 3.0]]])
  >>> converted = tf.image.rgb_to_grayscale(original)
  >>> print(converted.numpy())
  [[[1.81...]]]
  Args:
    images: The RGB tensor to convert. The last dimension must have size 3 and
      should contain RGB values.
    name: A name for the operation (optional).
  Returns:
    The converted grayscale image(s).
  """
  with ops.name_scope(name, 'rgb_to_grayscale', [images]) as name:
    images = ops.convert_to_tensor(images, name='images')
    # Remember original dtype to so we can convert back if needed
    orig_dtype = images.dtype
    flt_image = convert_image_dtype(images, dtypes.float32)
    # Reference for converting between RGB and grayscale.
    # https://en.wikipedia.org/wiki/Luma_%28video%29
    rgb_weights = [0.2989, 0.5870, 0.1140]
    gray_float = math_ops.tensordot(flt_image, rgb_weights, [-1, -1])
    gray_float = array_ops.expand_dims(gray_float, -1)
    return convert_image_dtype(gray_float, orig_dtype, name=name)
@tf_export('image.grayscale_to_rgb')
@dispatch.add_dispatch_support
def grayscale_to_rgb(images, name=None):
  """Converts one or more images from Grayscale to RGB.
  Outputs a tensor of the same `DType` and rank as `images`.  The size of the
  last dimension of the output is 3, containing the RGB value of the pixels.
  The input images' last dimension must be size 1.
  >>> original = tf.constant([[[1.0], [2.0], [3.0]]])
  >>> converted = tf.image.grayscale_to_rgb(original)
  >>> print(converted.numpy())
  [[[1. 1. 1.]
    [2. 2. 2.]
    [3. 3. 3.]]]
  Args:
    images: The Grayscale tensor to convert. The last dimension must be size 1.
    name: A name for the operation (optional).
  Returns:
    The converted grayscale image(s).
  """
  with ops.name_scope(name, 'grayscale_to_rgb', [images]) as name:
    images = _AssertGrayscaleImage(images)
    images = ops.convert_to_tensor(images, name='images')
    rank_1 = array_ops.expand_dims(array_ops.rank(images) - 1, 0)
    shape_list = ([array_ops.ones(rank_1, dtype=dtypes.int32)] +
                  [array_ops.expand_dims(3, 0)])
    multiples = array_ops.concat(shape_list, 0)
    rgb = array_ops.tile(images, multiples, name=name)
    rgb.set_shape(images.get_shape()[:-1].concatenate([3]))
    return rgb
# pylint: disable=invalid-name
@tf_export('image.random_hue')
@dispatch.add_dispatch_support
def random_hue(image, max_delta, seed=None):
  """Adjust the hue of RGB images by a random factor.
  Equivalent to `adjust_hue()` but uses a `delta` randomly
  picked in the interval `[-max_delta, max_delta)`.
  `max_delta` must be in the interval `[0, 0.5]`.
  Usage Example:
  >>> x = [[[1.0, 2.0, 3.0],
  ...       [4.0, 5.0, 6.0]],
  ...     [[7.0, 8.0, 9.0],
  ...       [10.0, 11.0, 12.0]]]
  >>> tf.image.random_hue(x, 0.2)
  <tf.Tensor: shape=(2, 2, 3), dtype=float32, numpy=...>
  For producing deterministic results given a `seed` value, use
  `tf.image.stateless_random_hue`. Unlike using the `seed` param with
  `tf.image.random_*` ops, `tf.image.stateless_random_*` ops guarantee the same
  results given the same seed independent of how many times the function is
  called, and independent of global seed settings (e.g. tf.random.set_seed).
  Args:
    image: RGB image or images. The size of the last dimension must be 3.
    max_delta: float. The maximum value for the random delta.
    seed: An operation-specific seed. It will be used in conjunction with the
      graph-level seed to determine the real seeds that will be used in this
      operation. Please see the documentation of set_random_seed for its
      interaction with the graph-level random seed.
  Returns:
    Adjusted image(s), same shape and DType as `image`.
  Raises:
    ValueError: if `max_delta` is invalid.
  """
  if max_delta > 0.5:
    raise ValueError('max_delta must be <= 0.5.')
  if max_delta < 0:
    raise ValueError('max_delta must be non-negative.')
  delta = random_ops.random_uniform([], -max_delta, max_delta, seed=seed)
  return adjust_hue(image, delta)
@tf_export('image.stateless_random_hue', v1=[])
@dispatch.add_dispatch_support
def stateless_random_hue(image, max_delta, seed):
  """Adjust the hue of RGB images by a random factor deterministically.
  Equivalent to `adjust_hue()` but uses a `delta` randomly picked in the
  interval `[-max_delta, max_delta)`.
  Guarantees the same results given the same `seed` independent of how many
  times the function is called, and independent of global seed settings (e.g.
  `tf.random.set_seed`).
  `max_delta` must be in the interval `[0, 0.5]`.
  Usage Example:
  >>> x = [[[1.0, 2.0, 3.0],
  ...       [4.0, 5.0, 6.0]],
  ...      [[7.0, 8.0, 9.0],
  ...       [10.0, 11.0, 12.0]]]
  >>> seed = (1, 2)
  >>> tf.image.stateless_random_hue(x, 0.2, seed)
  <tf.Tensor: shape=(2, 2, 3), dtype=float32, numpy=
  array([[[ 1.6514902,  1.       ,  3.       ],
          [ 4.65149  ,  4.       ,  6.       ]],
         [[ 7.65149  ,  7.       ,  9.       ],
          [10.65149  , 10.       , 12.       ]]], dtype=float32)>
  Args:
    image: RGB image or images. The size of the last dimension must be 3.
    max_delta: float. The maximum value for the random delta.
    seed: A shape [2] Tensor, the seed to the random number generator. Must have
      dtype `int32` or `int64`. (When using XLA, only `int32` is allowed.)
  Returns:
    Adjusted image(s), same shape and DType as `image`.
  Raises:
    ValueError: if `max_delta` is invalid.
  """
  if max_delta > 0.5:
    raise ValueError('max_delta must be <= 0.5.')
  if max_delta < 0:
    raise ValueError('max_delta must be non-negative.')
  delta = stateless_random_ops.stateless_random_uniform(
      shape=[], minval=-max_delta, maxval=max_delta, seed=seed)
  return adjust_hue(image, delta)
@tf_export('image.adjust_hue')
@dispatch.add_dispatch_support
def adjust_hue(image, delta, name=None):
  """Adjust hue of RGB images.
  This is a convenience method that converts an RGB image to float
  representation, converts it to HSV, adds an offset to the
  hue channel, converts back to RGB and then back to the original
  data type. If several adjustments are chained it is advisable to minimize
  the number of redundant conversions.
  `image` is an RGB image.  The image hue is adjusted by converting the
  image(s) to HSV and rotating the hue channel (H) by
  `delta`.  The image is then converted back to RGB.
  `delta` must be in the interval `[-1, 1]`.
  Usage Example:
  >>> x = [[[1.0, 2.0, 3.0],
  ...       [4.0, 5.0, 6.0]],
  ...     [[7.0, 8.0, 9.0],
  ...       [10.0, 11.0, 12.0]]]
  >>> tf.image.adjust_hue(x, 0.2)
  <tf.Tensor: shape=(2, 2, 3), dtype=float32, numpy=
  array([[[ 2.3999996,  1.       ,  3.       ],
          [ 5.3999996,  4.       ,  6.       ]],
        [[ 8.4      ,  7.       ,  9.       ],
          [11.4      , 10.       , 12.       ]]], dtype=float32)>
  Args:
    image: RGB image or images. The size of the last dimension must be 3.
    delta: float.  How much to add to the hue channel.
    name: A name for this operation (optional).
  Returns:
    Adjusted image(s), same shape and DType as `image`.
  Raises:
    InvalidArgumentError: image must have at least 3 dimensions.
    InvalidArgumentError: The size of the last dimension must be 3.
    ValueError: if `delta` is not in the interval of `[-1, 1]`.
  Usage Example:
  >>> image = [[[1, 2, 3], [4, 5, 6]],
  ...          [[7, 8, 9], [10, 11, 12]],
  ...          [[13, 14, 15], [16, 17, 18]]]
  >>> image = tf.constant(image)
  >>> tf.image.adjust_hue(image, 0.2)
  <tf.Tensor: shape=(3, 2, 3), dtype=int32, numpy=
  array([[[ 2,  1,  3],
        [ 5,  4,  6]],
       [[ 8,  7,  9],
        [11, 10, 12]],
       [[14, 13, 15],
        [17, 16, 18]]], dtype=int32)>
  """
  with ops.name_scope(name, 'adjust_hue', [image]) as name:
    if context.executing_eagerly():
      if delta < -1 or delta > 1:
        raise ValueError('delta must be in the interval [-1, 1]')
    image = ops.convert_to_tensor(image, name='image')
    # Remember original dtype to so we can convert back if needed
    orig_dtype = image.dtype
    if orig_dtype in (dtypes.float16, dtypes.float32):
      flt_image = image
    else:
      flt_image = convert_image_dtype(image, dtypes.float32)
    rgb_altered = gen_image_ops.adjust_hue(flt_image, delta)
    return convert_image_dtype(rgb_altered, orig_dtype)
# pylint: disable=invalid-name
@tf_export('image.random_jpeg_quality')
@dispatch.add_dispatch_support
def random_jpeg_quality(image, min_jpeg_quality, max_jpeg_quality, seed=None):
  """Randomly changes jpeg encoding quality for inducing jpeg noise.
  `min_jpeg_quality` must be in the interval `[0, 100]` and less than
  `max_jpeg_quality`.
  `max_jpeg_quality` must be in the interval `[0, 100]`.
  Usage Example:
  >>> x = tf.constant([[[1, 2, 3],
  ...                   [4, 5, 6]],
  ...                  [[7, 8, 9],
  ...                   [10, 11, 12]]], dtype=tf.uint8)
  >>> tf.image.random_jpeg_quality(x, 75, 95)
  <tf.Tensor: shape=(2, 2, 3), dtype=uint8, numpy=...>
  For producing deterministic results given a `seed` value, use
  `tf.image.stateless_random_jpeg_quality`. Unlike using the `seed` param
  with `tf.image.random_*` ops, `tf.image.stateless_random_*` ops guarantee the
  same results given the same seed independent of how many times the function is
  called, and independent of global seed settings (e.g. tf.random.set_seed).
  Args:
    image: 3D image. Size of the last dimension must be 1 or 3.
    min_jpeg_quality: Minimum jpeg encoding quality to use.
    max_jpeg_quality: Maximum jpeg encoding quality to use.
    seed: An operation-specific seed. It will be used in conjunction with the
      graph-level seed to determine the real seeds that will be used in this
      operation. Please see the documentation of set_random_seed for its
      interaction with the graph-level random seed.
  Returns:
    Adjusted image(s), same shape and DType as `image`.
  Raises:
    ValueError: if `min_jpeg_quality` or `max_jpeg_quality` is invalid.
  """
  if (min_jpeg_quality < 0 or max_jpeg_quality < 0 or min_jpeg_quality > 100 or
      max_jpeg_quality > 100):
    raise ValueError('jpeg encoding range must be between 0 and 100.')
  if min_jpeg_quality >= max_jpeg_quality:
    raise ValueError('`min_jpeg_quality` must be less than `max_jpeg_quality`.')
  jpeg_quality = random_ops.random_uniform([],
                                           min_jpeg_quality,
                                           max_jpeg_quality,
                                           seed=seed,
                                           dtype=dtypes.int32)
  return adjust_jpeg_quality(image, jpeg_quality)
@tf_export('image.stateless_random_jpeg_quality', v1=[])
@dispatch.add_dispatch_support
def stateless_random_jpeg_quality(image,
                                  min_jpeg_quality,
                                  max_jpeg_quality,
                                  seed):
  """Deterministically radomize jpeg encoding quality for inducing jpeg noise.
  Guarantees the same results given the same `seed` independent of how many
  times the function is called, and independent of global seed settings (e.g.
  `tf.random.set_seed`).
  `min_jpeg_quality` must be in the interval `[0, 100]` and less than
  `max_jpeg_quality`.
  `max_jpeg_quality` must be in the interval `[0, 100]`.
  Usage Example:
  >>> x = tf.constant([[[1, 2, 3],
  ...                   [4, 5, 6]],
  ...                  [[7, 8, 9],
  ...                   [10, 11, 12]]], dtype=tf.uint8)
  >>> seed = (1, 2)
  >>> tf.image.stateless_random_jpeg_quality(x, 75, 95, seed)
  <tf.Tensor: shape=(2, 2, 3), dtype=uint8, numpy=
  array([[[ 0,  4,  5],
          [ 1,  5,  6]],
         [[ 5,  9, 10],
          [ 5,  9, 10]]], dtype=uint8)>
  Args:
    image: 3D image. Size of the last dimension must be 1 or 3.
    min_jpeg_quality: Minimum jpeg encoding quality to use.
    max_jpeg_quality: Maximum jpeg encoding quality to use.
    seed: A shape [2] Tensor, the seed to the random number generator. Must have
      dtype `int32` or `int64`. (When using XLA, only `int32` is allowed.)
  Returns:
    Adjusted image(s), same shape and DType as `image`.
  Raises:
    ValueError: if `min_jpeg_quality` or `max_jpeg_quality` is invalid.
  """
  if (min_jpeg_quality < 0 or max_jpeg_quality < 0 or min_jpeg_quality > 100 or
      max_jpeg_quality > 100):
    raise ValueError('jpeg encoding range must be between 0 and 100.')
  if min_jpeg_quality >= max_jpeg_quality:
    raise ValueError('`min_jpeg_quality` must be less than `max_jpeg_quality`.')
  jpeg_quality = stateless_random_ops.stateless_random_uniform(
      shape=[], minval=min_jpeg_quality, maxval=max_jpeg_quality, seed=seed,
      dtype=dtypes.int32)
  return adjust_jpeg_quality(image, jpeg_quality)
@tf_export('image.adjust_jpeg_quality')
@dispatch.add_dispatch_support
def adjust_jpeg_quality(image, jpeg_quality, name=None):
  """Adjust jpeg encoding quality of an image.
  This is a convenience method that converts an image to uint8 representation,
  encodes it to jpeg with `jpeg_quality`, decodes it, and then converts back
  to the original data type.
  `jpeg_quality` must be in the interval `[0, 100]`.
  Usage Examples:
  >>> x = [[[0.01, 0.02, 0.03],
  ...       [0.04, 0.05, 0.06]],
  ...      [[0.07, 0.08, 0.09],
  ...       [0.10, 0.11, 0.12]]]
  >>> x_jpeg = tf.image.adjust_jpeg_quality(x, 75)
  >>> x_jpeg.numpy()
  array([[[0.00392157, 0.01960784, 0.03137255],
          [0.02745098, 0.04313726, 0.05490196]],
         [[0.05882353, 0.07450981, 0.08627451],
          [0.08235294, 0.09803922, 0.10980393]]], dtype=float32)
  Note that floating point values are expected to have values in the range
  [0,1) and values outside this range are clipped.
  >>> x = [[[1.0, 2.0, 3.0],
  ...       [4.0, 5.0, 6.0]],
  ...     [[7.0, 8.0, 9.0],
  ...       [10.0, 11.0, 12.0]]]
  >>> tf.image.adjust_jpeg_quality(x, 75)
  <tf.Tensor: shape=(2, 2, 3), dtype=float32, numpy=
  array([[[1., 1., 1.],
          [1., 1., 1.]],
         [[1., 1., 1.],
          [1., 1., 1.]]], dtype=float32)>
  Note that `jpeg_quality` 100 is still lossy compresson.
  >>> x = tf.constant([[[1, 2, 3],
  ...                   [4, 5, 6]],
  ...                  [[7, 8, 9],
  ...                   [10, 11, 12]]], dtype=tf.uint8)
  >>> tf.image.adjust_jpeg_quality(x, 100)
  <tf.Tensor: shape(2, 2, 3), dtype=uint8, numpy=
  array([[[ 0,  1,  3],
          [ 3,  4,  6]],
         [[ 6,  7,  9],
          [ 9, 10, 12]]], dtype=uint8)>
  Args:
    image: 3D image. The size of the last dimension must be None, 1 or 3.
    jpeg_quality: Python int or Tensor of type int32. jpeg encoding quality.
    name: A name for this operation (optional).
  Returns:
    Adjusted image, same shape and DType as `image`.
  Raises:
    InvalidArgumentError: quality must be in [0,100]
    InvalidArgumentError: image must have 1 or 3 channels
  """
  with ops.name_scope(name, 'adjust_jpeg_quality', [image]):
    image = ops.convert_to_tensor(image, name='image')
    channels = image.shape.as_list()[-1]
    # Remember original dtype to so we can convert back if needed
    orig_dtype = image.dtype
    image = convert_image_dtype(image, dtypes.uint8, saturate=True)
    if not _is_tensor(jpeg_quality):
      # If jpeg_quality is a int (not tensor).
      jpeg_quality = ops.convert_to_tensor(jpeg_quality, dtype=dtypes.int32)
    image = gen_image_ops.encode_jpeg_variable_quality(image, jpeg_quality)
    image = gen_image_ops.decode_jpeg(image, channels=channels)
    return convert_image_dtype(image, orig_dtype, saturate=True)
@tf_export('image.random_saturation')
@dispatch.add_dispatch_support
def random_saturation(image, lower, upper, seed=None):
  """Adjust the saturation of RGB images by a random factor.
  Equivalent to `adjust_saturation()` but uses a `saturation_factor` randomly
  picked in the interval `[lower, upper)`.
  Usage Example:
  >>> x = [[[1.0, 2.0, 3.0],
  ...       [4.0, 5.0, 6.0]],
  ...     [[7.0, 8.0, 9.0],
  ...       [10.0, 11.0, 12.0]]]
  >>> tf.image.random_saturation(x, 5, 10)
  <tf.Tensor: shape=(2, 2, 3), dtype=float32, numpy=
  array([[[ 0. ,  1.5,  3. ],
          [ 0. ,  3. ,  6. ]],
         [[ 0. ,  4.5,  9. ],
          [ 0. ,  6. , 12. ]]], dtype=float32)>
  For producing deterministic results given a `seed` value, use
  `tf.image.stateless_random_saturation`. Unlike using the `seed` param
  with `tf.image.random_*` ops, `tf.image.stateless_random_*` ops guarantee the
  same results given the same seed independent of how many times the function is
  called, and independent of global seed settings (e.g. tf.random.set_seed).
  Args:
    image: RGB image or images. The size of the last dimension must be 3.
    lower: float.  Lower bound for the random saturation factor.
    upper: float.  Upper bound for the random saturation factor.
    seed: An operation-specific seed. It will be used in conjunction with the
      graph-level seed to determine the real seeds that will be used in this
      operation. Please see the documentation of set_random_seed for its
      interaction with the graph-level random seed.
  Returns:
    Adjusted image(s), same shape and DType as `image`.
  Raises:
    ValueError: if `upper <= lower` or if `lower < 0`.
  """
  if upper <= lower:
    raise ValueError('upper must be > lower.')
  if lower < 0:
    raise ValueError('lower must be non-negative.')
  saturation_factor = random_ops.random_uniform([], lower, upper, seed=seed)
  return adjust_saturation(image, saturation_factor)
@tf_export('image.stateless_random_saturation', v1=[])
@dispatch.add_dispatch_support
def stateless_random_saturation(image, lower, upper, seed=None):
  """Adjust the saturation of RGB images by a random factor deterministically.
  Equivalent to `adjust_saturation()` but uses a `saturation_factor` randomly
  picked in the interval `[lower, upper)`.
  Guarantees the same results given the same `seed` independent of how many
  times the function is called, and independent of global seed settings (e.g.
  `tf.random.set_seed`).
  Usage Example:
  >>> x = [[[1.0, 2.0, 3.0],
  ...       [4.0, 5.0, 6.0]],
  ...      [[7.0, 8.0, 9.0],
  ...       [10.0, 11.0, 12.0]]]
  >>> seed = (1, 2)
  >>> tf.image.stateless_random_saturation(x, 0.5, 1.0, seed)
  <tf.Tensor: shape=(2, 2, 3), dtype=float32, numpy=
  array([[[ 1.1559395,  2.0779698,  3.       ],
          [ 4.1559396,  5.07797  ,  6.       ]],
         [[ 7.1559396,  8.07797  ,  9.       ],
          [10.155939 , 11.07797  , 12.       ]]], dtype=float32)>
  Args:
    image: RGB image or images. The size of the last dimension must be 3.
    lower: float.  Lower bound for the random saturation factor.
    upper: float.  Upper bound for the random saturation factor.
    seed: A shape [2] Tensor, the seed to the random number generator. Must have
      dtype `int32` or `int64`. (When using XLA, only `int32` is allowed.)
  Returns:
    Adjusted image(s), same shape and DType as `image`.
  Raises:
    ValueError: if `upper <= lower` or if `lower < 0`.
  """
  if upper <= lower:
    raise ValueError('upper must be > lower.')
  if lower < 0:
    raise ValueError('lower must be non-negative.')
  saturation_factor = stateless_random_ops.stateless_random_uniform(
      shape=[], minval=lower, maxval=upper, seed=seed)
  return adjust_saturation(image, saturation_factor)
@tf_export('image.adjust_saturation')
@dispatch.add_dispatch_support
def adjust_saturation(image, saturation_factor, name=None):
  """Adjust saturation of RGB images.
  This is a convenience method that converts RGB images to float
  representation, converts them to HSV, adds an offset to the
  saturation channel, converts back to RGB and then back to the original
  data type. If several adjustments are chained it is advisable to minimize
  the number of redundant conversions.
  `image` is an RGB image or images.  The image saturation is adjusted by
  converting the images to HSV and multiplying the saturation (S) channel by
  `saturation_factor` and clipping. The images are then converted back to RGB.
  `saturation_factor` must be in the interval `[0, inf)`.
  Usage Example:
  >>> x = [[[1.0, 2.0, 3.0],
  ...       [4.0, 5.0, 6.0]],
  ...     [[7.0, 8.0, 9.0],
  ...       [10.0, 11.0, 12.0]]]
  >>> tf.image.adjust_saturation(x, 0.5)
  <tf.Tensor: shape=(2, 2, 3), dtype=float32, numpy=
  array([[[ 2. ,  2.5,  3. ],
          [ 5. ,  5.5,  6. ]],
         [[ 8. ,  8.5,  9. ],
          [11. , 11.5, 12. ]]], dtype=float32)>
  Args:
    image: RGB image or images. The size of the last dimension must be 3.
    saturation_factor: float. Factor to multiply the saturation by.
    name: A name for this operation (optional).
  Returns:
    Adjusted image(s), same shape and DType as `image`.
  Raises:
    InvalidArgumentError: input must have 3 channels
  """
  with ops.name_scope(name, 'adjust_saturation', [image]) as name:
    image = ops.convert_to_tensor(image, name='image')
    # Remember original dtype to so we can convert back if needed
    orig_dtype = image.dtype
    if orig_dtype in (dtypes.float16, dtypes.float32):
      flt_image = image
    else:
      flt_image = convert_image_dtype(image, dtypes.float32)
    adjusted = gen_image_ops.adjust_saturation(flt_image, saturation_factor)
    return convert_image_dtype(adjusted, orig_dtype)
@tf_export('io.is_jpeg', 'image.is_jpeg', v1=['io.is_jpeg', 'image.is_jpeg'])
def is_jpeg(contents, name=None):
  r"""Convenience function to check if the 'contents' encodes a JPEG image.
  Args:
    contents: 0-D `string`. The encoded image bytes.
    name: A name for the operation (optional)
  Returns:
     A scalar boolean tensor indicating if 'contents' may be a JPEG image.
     is_jpeg is susceptible to false positives.
  """
  # Normal JPEGs start with \xff\xd8\xff\xe0
  # JPEG with EXIF starts with \xff\xd8\xff\xe1
  # Use \xff\xd8\xff to cover both.
  with ops.name_scope(name, 'is_jpeg'):
    substr = string_ops.substr(contents, 0, 3)
    return math_ops.equal(substr, b'\xff\xd8\xff', name=name)
def _is_png(contents, name=None):
  r"""Convenience function to check if the 'contents' encodes a PNG image.
  Args:
    contents: 0-D `string`. The encoded image bytes.
    name: A name for the operation (optional)
  Returns:
     A scalar boolean tensor indicating if 'contents' may be a PNG image.
     is_png is susceptible to false positives.
  """
  with ops.name_scope(name, 'is_png'):
    substr = string_ops.substr(contents, 0, 3)
    return math_ops.equal(substr, b'\211PN', name=name)
tf_export(
    'io.decode_and_crop_jpeg',
    'image.decode_and_crop_jpeg',
    v1=['io.decode_and_crop_jpeg', 'image.decode_and_crop_jpeg'])(
        dispatch.add_dispatch_support(gen_image_ops.decode_and_crop_jpeg))
tf_export(
    'io.decode_bmp',
    'image.decode_bmp',
    v1=['io.decode_bmp', 'image.decode_bmp'])(
        dispatch.add_dispatch_support(gen_image_ops.decode_bmp))
tf_export(
    'io.decode_gif',
    'image.decode_gif',
    v1=['io.decode_gif', 'image.decode_gif'])(
        dispatch.add_dispatch_support(gen_image_ops.decode_gif))
tf_export(
    'io.decode_jpeg',
    'image.decode_jpeg',
    v1=['io.decode_jpeg', 'image.decode_jpeg'])(
        dispatch.add_dispatch_support(gen_image_ops.decode_jpeg))
tf_export(
    'io.decode_png',
    'image.decode_png',
    v1=['io.decode_png', 'image.decode_png'])(
        dispatch.add_dispatch_support(gen_image_ops.decode_png))
tf_export(
    'io.encode_jpeg',
    'image.encode_jpeg',
    v1=['io.encode_jpeg', 'image.encode_jpeg'])(
        dispatch.add_dispatch_support(gen_image_ops.encode_jpeg))
tf_export(
    'io.extract_jpeg_shape',
    'image.extract_jpeg_shape',
    v1=['io.extract_jpeg_shape', 'image.extract_jpeg_shape'])(
        dispatch.add_dispatch_support(gen_image_ops.extract_jpeg_shape))
@tf_export('io.encode_png', 'image.encode_png')
@dispatch.add_dispatch_support
def encode_png(image, compression=-1, name=None):
  r"""PNG-encode an image.
  `image` is a rank-N Tensor of type uint8 or uint16 with shape `batch_dims +
  [height, width, channels]`, where `channels` is:
  *   1: for grayscale.
  *   2: for grayscale + alpha.
  *   3: for RGB.
  *   4: for RGBA.
  The ZLIB compression level, `compression`, can be -1 for the PNG-encoder
  default or a value from 0 to 9.  9 is the highest compression level,
  generating the smallest output, but is slower.
  Args:
    image: A `Tensor`. Must be one of the following types: `uint8`, `uint16`.
      Rank N >= 3 with shape `batch_dims + [height, width, channels]`.
    compression: An optional `int`. Defaults to `-1`. Compression level.
    name: A name for the operation (optional).
  Returns:
    A `Tensor` of type `string`.
  """
  return gen_image_ops.encode_png(
      ops.convert_to_tensor(image), compression, name)
