"/home/cc/Workspace/tfconstraint/python/ops/image_ops_impl.py"
@tf_export(
    'io.decode_image',
    'image.decode_image',
    v1=['io.decode_image', 'image.decode_image'])
@dispatch.add_dispatch_support
def decode_image(contents,
                 channels=None,
                 dtype=dtypes.uint8,
                 name=None,
                 expand_animations=True):
  """Function for `decode_bmp`, `decode_gif`, `decode_jpeg`, and `decode_png`.
  Detects whether an image is a BMP, GIF, JPEG, or PNG, and performs the
  appropriate operation to convert the input bytes `string` into a `Tensor`
  of type `dtype`.
  Note: `decode_gif` returns a 4-D array `[num_frames, height, width, 3]`, as
  opposed to `decode_bmp`, `decode_jpeg` and `decode_png`, which return 3-D
  arrays `[height, width, num_channels]`. Make sure to take this into account
  when constructing your graph if you are intermixing GIF files with BMP, JPEG,
  and/or PNG files. Alternately, set the `expand_animations` argument of this
  function to `False`, in which case the op will return 3-dimensional tensors
  and will truncate animated GIF files to the first frame.
  NOTE: If the first frame of an animated GIF does not occupy the entire
  canvas (maximum frame width x maximum frame height), then it fills the
  unoccupied areas (in the first frame) with zeros (black). For frames after the
  first frame that does not occupy the entire canvas, it uses the previous
  frame to fill the unoccupied areas.
  Args:
    contents: A `Tensor` of type `string`. 0-D. The encoded image bytes.
    channels: An optional `int`. Defaults to `0`. Number of color channels for
      the decoded image.
    dtype: The desired DType of the returned `Tensor`.
    name: A name for the operation (optional)
    expand_animations: An optional `bool`. Defaults to `True`. Controls the
      shape of the returned op's output. If `True`, the returned op will produce
      a 3-D tensor for PNG, JPEG, and BMP files; and a 4-D tensor for all GIFs,
      whether animated or not. If, `False`, the returned op will produce a 3-D
      tensor for all file types and will truncate animated GIFs to the first
      frame.
  Returns:
    `Tensor` with type `dtype` and a 3- or 4-dimensional shape, depending on
    the file type and the value of the `expand_animations` parameter.
  Raises:
    ValueError: On incorrect number of channels.
  """
  with ops.name_scope(name, 'decode_image'):
    channels = 0 if channels is None else channels
    if dtype not in [dtypes.float32, dtypes.uint8, dtypes.uint16]:
      dest_dtype = dtype
      dtype = dtypes.uint16
      return convert_image_dtype(
          gen_image_ops.decode_image(
              contents=contents,
              channels=channels,
              expand_animations=expand_animations,
              dtype=dtype), dest_dtype)
    else:
      return gen_image_ops.decode_image(
          contents=contents,
          channels=channels,
          expand_animations=expand_animations,
          dtype=dtype)
@tf_export('image.total_variation')
@dispatch.add_dispatch_support
def total_variation(images, name=None):
  """Calculate and return the total variation for one or more images.
  The total variation is the sum of the absolute differences for neighboring
  pixel-values in the input images. This measures how much noise is in the
  images.
  This can be used as a loss-function during optimization so as to suppress
  noise in images. If you have a batch of images, then you should calculate
  the scalar loss-value as the sum:
  `loss = tf.reduce_sum(tf.image.total_variation(images))`
  This implements the anisotropic 2-D version of the formula described here:
  https://en.wikipedia.org/wiki/Total_variation_denoising
  Args:
    images: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor
      of shape `[height, width, channels]`.
    name: A name for the operation (optional).
  Raises:
    ValueError: if images.shape is not a 3-D or 4-D vector.
  Returns:
    The total variation of `images`.
    If `images` was 4-D, return a 1-D float Tensor of shape `[batch]` with the
    total variation for each image in the batch.
    If `images` was 3-D, return a scalar float with the total variation for
    that image.
  """
  with ops.name_scope(name, 'total_variation'):
    ndims = images.get_shape().ndims
    if ndims == 3:
      # The input is a single image with shape [height, width, channels].
      # Calculate the difference of neighboring pixel-values.
      # The images are shifted one pixel along the height and width by slicing.
      pixel_dif1 = images[1:, :, :] - images[:-1, :, :]
      pixel_dif2 = images[:, 1:, :] - images[:, :-1, :]
      # Sum for all axis. (None is an alias for all axis.)
      sum_axis = None
    elif ndims == 4:
      # The input is a batch of images with shape:
      # [batch, height, width, channels].
      # Calculate the difference of neighboring pixel-values.
      # The images are shifted one pixel along the height and width by slicing.
      pixel_dif1 = images[:, 1:, :, :] - images[:, :-1, :, :]
      pixel_dif2 = images[:, :, 1:, :] - images[:, :, :-1, :]
      # Only sum for the last 3 axis.
      # This results in a 1-D tensor with the total variation for each image.
      sum_axis = [1, 2, 3]
    else:
      raise ValueError('\'images\' must be either 3 or 4-dimensional.')
    # Calculate the total variation by taking the absolute value of the
    # pixel-differences and summing over the appropriate axis.
    tot_var = (
        math_ops.reduce_sum(math_ops.abs(pixel_dif1), axis=sum_axis) +
        math_ops.reduce_sum(math_ops.abs(pixel_dif2), axis=sum_axis))
  return tot_var
@tf_export('image.sample_distorted_bounding_box', v1=[])
@dispatch.add_dispatch_support
def sample_distorted_bounding_box_v2(image_size,
                                     bounding_boxes,
                                     seed=0,
                                     min_object_covered=0.1,
                                     aspect_ratio_range=None,
                                     area_range=None,
                                     max_attempts=None,
                                     use_image_if_no_bounding_boxes=None,
                                     name=None):
  """Generate a single randomly distorted bounding box for an image.
  Bounding box annotations are often supplied in addition to ground-truth labels
  in image recognition or object localization tasks. A common technique for
  training such a system is to randomly distort an image while preserving
  its content, i.e. *data augmentation*. This Op outputs a randomly distorted
  localization of an object, i.e. bounding box, given an `image_size`,
  `bounding_boxes` and a series of constraints.
  The output of this Op is a single bounding box that may be used to crop the
  original image. The output is returned as 3 tensors: `begin`, `size` and
  `bboxes`. The first 2 tensors can be fed directly into `tf.slice` to crop the
  image. The latter may be supplied to `tf.image.draw_bounding_boxes` to
  visualize what the bounding box looks like.
  Bounding boxes are supplied and returned as `[y_min, x_min, y_max, x_max]`.
  The bounding box coordinates are floats in `[0.0, 1.0]` relative to the width
  and the height of the underlying image.
  For example,
  ```python
      # Generate a single distorted bounding box.
      begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
          tf.shape(image),
          bounding_boxes=bounding_boxes,
          min_object_covered=0.1)
      # Draw the bounding box in an image summary.
      image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0),
                                                    bbox_for_draw)
      tf.compat.v1.summary.image('images_with_box', image_with_box)
      # Employ the bounding box to distort the image.
      distorted_image = tf.slice(image, begin, size)
  ```
  Note that if no bounding box information is available, setting
  `use_image_if_no_bounding_boxes = true` will assume there is a single implicit
  bounding box covering the whole image. If `use_image_if_no_bounding_boxes` is
  false and no bounding boxes are supplied, an error is raised.
  For producing deterministic results given a `seed` value, use
  `tf.image.stateless_sample_distorted_bounding_box`. Unlike using the `seed`
  param with `tf.image.random_*` ops, `tf.image.stateless_random_*` ops
  guarantee the same results given the same seed independent of how many times
  the function is called, and independent of global seed settings
  (e.g. tf.random.set_seed).
  Args:
    image_size: A `Tensor`. Must be one of the following types: `uint8`, `int8`,
      `int16`, `int32`, `int64`. 1-D, containing `[height, width, channels]`.
    bounding_boxes: A `Tensor` of type `float32`. 3-D with shape `[batch, N, 4]`
      describing the N bounding boxes associated with the image.
    seed: An optional `int`. Defaults to `0`. If `seed` is set to non-zero, the
      random number generator is seeded by the given `seed`.  Otherwise, it is
      seeded by a random seed.
    min_object_covered: A Tensor of type `float32`. Defaults to `0.1`. The
      cropped area of the image must contain at least this fraction of any
      bounding box supplied. The value of this parameter should be non-negative.
      In the case of 0, the cropped area does not need to overlap any of the
      bounding boxes supplied.
    aspect_ratio_range: An optional list of `floats`. Defaults to `[0.75,
      1.33]`. The cropped area of the image must have an aspect `ratio = width /
      height` within this range.
    area_range: An optional list of `floats`. Defaults to `[0.05, 1]`. The
      cropped area of the image must contain a fraction of the supplied image
      within this range.
    max_attempts: An optional `int`. Defaults to `100`. Number of attempts at
      generating a cropped region of the image of the specified constraints.
      After `max_attempts` failures, return the entire image.
    use_image_if_no_bounding_boxes: An optional `bool`. Defaults to `False`.
      Controls behavior if no bounding boxes supplied. If true, assume an
      implicit bounding box covering the whole input. If false, raise an error.
    name: A name for the operation (optional).
  Returns:
    A tuple of `Tensor` objects (begin, size, bboxes).
    begin: A `Tensor`. Has the same type as `image_size`. 1-D, containing
    `[offset_height, offset_width, 0]`. Provide as input to
      `tf.slice`.
    size: A `Tensor`. Has the same type as `image_size`. 1-D, containing
    `[target_height, target_width, -1]`. Provide as input to
      `tf.slice`.
    bboxes: A `Tensor` of type `float32`. 3-D with shape `[1, 1, 4]` containing
    the distorted bounding box.
    Provide as input to `tf.image.draw_bounding_boxes`.
  Raises:
    ValueError: If no seed is specified and op determinism is enabled.
  """
  if seed:
    seed1, seed2 = random_seed.get_seed(seed)
  else:
    if config.is_op_determinism_enabled():
      raise ValueError(
          f'tf.image.sample_distorted_bounding_box requires a non-zero seed to '
          f'be passed in when determinism is enabled, but got seed={seed}. '
          f'Please pass in a non-zero seed, e.g. by passing "seed=1".')
    seed1, seed2 = (0, 0)
  with ops.name_scope(name, 'sample_distorted_bounding_box'):
    return gen_image_ops.sample_distorted_bounding_box_v2(
        image_size,
        bounding_boxes,
        seed=seed1,
        seed2=seed2,
        min_object_covered=min_object_covered,
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        max_attempts=max_attempts,
        use_image_if_no_bounding_boxes=use_image_if_no_bounding_boxes,
        name=name)
@tf_export('image.stateless_sample_distorted_bounding_box', v1=[])
@dispatch.add_dispatch_support
def stateless_sample_distorted_bounding_box(image_size,
                                            bounding_boxes,
                                            seed,
                                            min_object_covered=0.1,
                                            aspect_ratio_range=None,
                                            area_range=None,
                                            max_attempts=None,
                                            use_image_if_no_bounding_boxes=None,
                                            name=None):
  """Generate a randomly distorted bounding box for an image deterministically.
  Bounding box annotations are often supplied in addition to ground-truth labels
  in image recognition or object localization tasks. A common technique for
  training such a system is to randomly distort an image while preserving
  its content, i.e. *data augmentation*. This Op, given the same `seed`,
  deterministically outputs a randomly distorted localization of an object, i.e.
  bounding box, given an `image_size`, `bounding_boxes` and a series of
  constraints.
  The output of this Op is a single bounding box that may be used to crop the
  original image. The output is returned as 3 tensors: `begin`, `size` and
  `bboxes`. The first 2 tensors can be fed directly into `tf.slice` to crop the
  image. The latter may be supplied to `tf.image.draw_bounding_boxes` to
  visualize what the bounding box looks like.
  Bounding boxes are supplied and returned as `[y_min, x_min, y_max, x_max]`.
  The bounding box coordinates are floats in `[0.0, 1.0]` relative to the width
  and the height of the underlying image.
  The output of this Op is guaranteed to be the same given the same `seed` and
  is independent of how many times the function is called, and independent of
  global seed settings (e.g. `tf.random.set_seed`).
  Example usage:
  >>> image = np.array([[[1], [2], [3]], [[4], [5], [6]], [[7], [8], [9]]])
  >>> bbox = tf.constant(
  ...   [0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
  >>> seed = (1, 2)
  >>> # Generate a single distorted bounding box.
  >>> bbox_begin, bbox_size, bbox_draw = (
  ...   tf.image.stateless_sample_distorted_bounding_box(
  ...     tf.shape(image), bounding_boxes=bbox, seed=seed))
  >>> # Employ the bounding box to distort the image.
  >>> tf.slice(image, bbox_begin, bbox_size)
  <tf.Tensor: shape=(2, 2, 1), dtype=int64, numpy=
  array([[[1],
          [2]],
         [[4],
          [5]]])>
  >>> # Draw the bounding box in an image summary.
  >>> colors = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
  >>> tf.image.draw_bounding_boxes(
  ...   tf.expand_dims(tf.cast(image, tf.float32),0), bbox_draw, colors)
  <tf.Tensor: shape=(1, 3, 3, 1), dtype=float32, numpy=
  array([[[[1.],
           [1.],
           [3.]],
          [[1.],
           [1.],
           [6.]],
          [[7.],
           [8.],
           [9.]]]], dtype=float32)>
  Note that if no bounding box information is available, setting
  `use_image_if_no_bounding_boxes = true` will assume there is a single implicit
  bounding box covering the whole image. If `use_image_if_no_bounding_boxes` is
  false and no bounding boxes are supplied, an error is raised.
  Args:
    image_size: A `Tensor`. Must be one of the following types: `uint8`, `int8`,
      `int16`, `int32`, `int64`. 1-D, containing `[height, width, channels]`.
    bounding_boxes: A `Tensor` of type `float32`. 3-D with shape `[batch, N, 4]`
      describing the N bounding boxes associated with the image.
    seed: A shape [2] Tensor, the seed to the random number generator. Must have
      dtype `int32` or `int64`. (When using XLA, only `int32` is allowed.)
    min_object_covered: A Tensor of type `float32`. Defaults to `0.1`. The
      cropped area of the image must contain at least this fraction of any
      bounding box supplied. The value of this parameter should be non-negative.
      In the case of 0, the cropped area does not need to overlap any of the
      bounding boxes supplied.
    aspect_ratio_range: An optional list of `floats`. Defaults to `[0.75,
      1.33]`. The cropped area of the image must have an aspect `ratio = width /
      height` within this range.
    area_range: An optional list of `floats`. Defaults to `[0.05, 1]`. The
      cropped area of the image must contain a fraction of the supplied image
      within this range.
    max_attempts: An optional `int`. Defaults to `100`. Number of attempts at
      generating a cropped region of the image of the specified constraints.
      After `max_attempts` failures, return the entire image.
    use_image_if_no_bounding_boxes: An optional `bool`. Defaults to `False`.
      Controls behavior if no bounding boxes supplied. If true, assume an
      implicit bounding box covering the whole input. If false, raise an error.
    name: A name for the operation (optional).
  Returns:
    A tuple of `Tensor` objects (begin, size, bboxes).
    begin: A `Tensor`. Has the same type as `image_size`. 1-D, containing
    `[offset_height, offset_width, 0]`. Provide as input to
      `tf.slice`.
    size: A `Tensor`. Has the same type as `image_size`. 1-D, containing
    `[target_height, target_width, -1]`. Provide as input to
      `tf.slice`.
    bboxes: A `Tensor` of type `float32`. 3-D with shape `[1, 1, 4]` containing
    the distorted bounding box.
    Provide as input to `tf.image.draw_bounding_boxes`.
  """
  with ops.name_scope(name, 'stateless_sample_distorted_bounding_box'):
    return gen_image_ops.stateless_sample_distorted_bounding_box(
        image_size=image_size,
        bounding_boxes=bounding_boxes,
        seed=seed,
        min_object_covered=min_object_covered,
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        max_attempts=max_attempts,
        use_image_if_no_bounding_boxes=use_image_if_no_bounding_boxes,
        name=name)
@tf_export(v1=['image.sample_distorted_bounding_box'])
@dispatch.add_dispatch_support
@deprecation.deprecated(
    date=None,
    instructions='`seed2` arg is deprecated.'
    'Use sample_distorted_bounding_box_v2 instead.')
def sample_distorted_bounding_box(image_size,
                                  bounding_boxes,
                                  seed=None,
                                  seed2=None,
                                  min_object_covered=0.1,
                                  aspect_ratio_range=None,
                                  area_range=None,
                                  max_attempts=None,
                                  use_image_if_no_bounding_boxes=None,
                                  name=None):
  """Generate a single randomly distorted bounding box for an image.
  Bounding box annotations are often supplied in addition to ground-truth labels
  in image recognition or object localization tasks. A common technique for
  training such a system is to randomly distort an image while preserving
  its content, i.e. *data augmentation*. This Op outputs a randomly distorted
  localization of an object, i.e. bounding box, given an `image_size`,
  `bounding_boxes` and a series of constraints.
  The output of this Op is a single bounding box that may be used to crop the
  original image. The output is returned as 3 tensors: `begin`, `size` and
  `bboxes`. The first 2 tensors can be fed directly into `tf.slice` to crop the
  image. The latter may be supplied to `tf.image.draw_bounding_boxes` to
  visualize what the bounding box looks like.
  Bounding boxes are supplied and returned as `[y_min, x_min, y_max, x_max]`.
  The
  bounding box coordinates are floats in `[0.0, 1.0]` relative to the width and
  height of the underlying image.
  For example,
  ```python
      # Generate a single distorted bounding box.
      begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
          tf.shape(image),
          bounding_boxes=bounding_boxes,
          min_object_covered=0.1)
      # Draw the bounding box in an image summary.
      image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0),
                                                    bbox_for_draw)
      tf.compat.v1.summary.image('images_with_box', image_with_box)
      # Employ the bounding box to distort the image.
      distorted_image = tf.slice(image, begin, size)
  ```
  Note that if no bounding box information is available, setting
  `use_image_if_no_bounding_boxes = True` will assume there is a single implicit
  bounding box covering the whole image. If `use_image_if_no_bounding_boxes` is
  false and no bounding boxes are supplied, an error is raised.
  Args:
    image_size: A `Tensor`. Must be one of the following types: `uint8`, `int8`,
      `int16`, `int32`, `int64`. 1-D, containing `[height, width, channels]`.
    bounding_boxes: A `Tensor` of type `float32`. 3-D with shape `[batch, N, 4]`
      describing the N bounding boxes associated with the image.
    seed: An optional `int`. Defaults to `0`. If either `seed` or `seed2` are
      set to non-zero, the random number generator is seeded by the given
      `seed`.  Otherwise, it is seeded by a random seed.
    seed2: An optional `int`. Defaults to `0`. A second seed to avoid seed
      collision.
    min_object_covered: A Tensor of type `float32`. Defaults to `0.1`. The
      cropped area of the image must contain at least this fraction of any
      bounding box supplied. The value of this parameter should be non-negative.
      In the case of 0, the cropped area does not need to overlap any of the
      bounding boxes supplied.
    aspect_ratio_range: An optional list of `floats`. Defaults to `[0.75,
      1.33]`. The cropped area of the image must have an aspect ratio = width /
      height within this range.
    area_range: An optional list of `floats`. Defaults to `[0.05, 1]`. The
      cropped area of the image must contain a fraction of the supplied image
      within this range.
    max_attempts: An optional `int`. Defaults to `100`. Number of attempts at
      generating a cropped region of the image of the specified constraints.
      After `max_attempts` failures, return the entire image.
    use_image_if_no_bounding_boxes: An optional `bool`. Defaults to `False`.
      Controls behavior if no bounding boxes supplied. If true, assume an
      implicit bounding box covering the whole input. If false, raise an error.
    name: A name for the operation (optional).
  Returns:
    A tuple of `Tensor` objects (begin, size, bboxes).
    begin: A `Tensor`. Has the same type as `image_size`. 1-D, containing
    `[offset_height, offset_width, 0]`. Provide as input to
      `tf.slice`.
    size: A `Tensor`. Has the same type as `image_size`. 1-D, containing
    `[target_height, target_width, -1]`. Provide as input to
      `tf.slice`.
    bboxes: A `Tensor` of type `float32`. 3-D with shape `[1, 1, 4]` containing
    the distorted bounding box.
      Provide as input to `tf.image.draw_bounding_boxes`.
  Raises:
    ValueError: If no seed is specified and op determinism is enabled.
  """
  if not seed and not seed2 and config.is_op_determinism_enabled():
    raise ValueError(
        f'tf.compat.v1.image.sample_distorted_bounding_box requires "seed" or '
        f'"seed2" to be non-zero when determinism is enabled. Please pass in '
        f'a non-zero seed, e.g. by passing "seed=1". Got seed={seed} and '
        f"seed2={seed2}")
  with ops.name_scope(name, 'sample_distorted_bounding_box'):
    return gen_image_ops.sample_distorted_bounding_box_v2(
        image_size,
        bounding_boxes,
        seed=seed,
        seed2=seed2,
        min_object_covered=min_object_covered,
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        max_attempts=max_attempts,
        use_image_if_no_bounding_boxes=use_image_if_no_bounding_boxes,
        name=name)
@tf_export('image.non_max_suppression')
@dispatch.add_dispatch_support
def non_max_suppression(boxes,
                        scores,
                        max_output_size,
                        iou_threshold=0.5,
                        score_threshold=float('-inf'),
                        name=None):
  """Greedily selects a subset of bounding boxes in descending order of score.
  Prunes away boxes that have high intersection-over-union (IOU) overlap
  with previously selected boxes.  Bounding boxes are supplied as
  `[y1, x1, y2, x2]`, where `(y1, x1)` and `(y2, x2)` are the coordinates of any
  diagonal pair of box corners and the coordinates can be provided as normalized
  (i.e., lying in the interval `[0, 1]`) or absolute.  Note that this algorithm
  is agnostic to where the origin is in the coordinate system.  Note that this
  algorithm is invariant to orthogonal transformations and translations
  of the coordinate system; thus translating or reflections of the coordinate
  system result in the same boxes being selected by the algorithm.
  The output of this operation is a set of integers indexing into the input
  collection of bounding boxes representing the selected boxes.  The bounding
  box coordinates corresponding to the selected indices can then be obtained
  using the `tf.gather` operation.  For example:
    ```python
    selected_indices = tf.image.non_max_suppression(
        boxes, scores, max_output_size, iou_threshold)
    selected_boxes = tf.gather(boxes, selected_indices)
    ```
  Args:
    boxes: A 2-D float `Tensor` of shape `[num_boxes, 4]`.
    scores: A 1-D float `Tensor` of shape `[num_boxes]` representing a single
      score corresponding to each box (each row of boxes).
    max_output_size: A scalar integer `Tensor` representing the maximum number
      of boxes to be selected by non-max suppression.
    iou_threshold: A 0-D float tensor representing the threshold for deciding
      whether boxes overlap too much with respect to IOU.
    score_threshold: A 0-D float tensor representing the threshold for deciding
      when to remove boxes based on score.
    name: A name for the operation (optional).
  Returns:
    selected_indices: A 1-D integer `Tensor` of shape `[M]` representing the
      selected indices from the boxes tensor, where `M <= max_output_size`.
  """
  with ops.name_scope(name, 'non_max_suppression'):
    iou_threshold = ops.convert_to_tensor(iou_threshold, name='iou_threshold')
    score_threshold = ops.convert_to_tensor(
        score_threshold, name='score_threshold')
    return gen_image_ops.non_max_suppression_v3(boxes, scores, max_output_size,
                                                iou_threshold, score_threshold)
@tf_export('image.non_max_suppression_with_scores')
@dispatch.add_dispatch_support
def non_max_suppression_with_scores(boxes,
                                    scores,
                                    max_output_size,
                                    iou_threshold=0.5,
                                    score_threshold=float('-inf'),
                                    soft_nms_sigma=0.0,
                                    name=None):
  """Greedily selects a subset of bounding boxes in descending order of score.
  Prunes away boxes that have high intersection-over-union (IOU) overlap
  with previously selected boxes.  Bounding boxes are supplied as
  `[y1, x1, y2, x2]`, where `(y1, x1)` and `(y2, x2)` are the coordinates of any
  diagonal pair of box corners and the coordinates can be provided as normalized
  (i.e., lying in the interval `[0, 1]`) or absolute.  Note that this algorithm
  is agnostic to where the origin is in the coordinate system.  Note that this
  algorithm is invariant to orthogonal transformations and translations
  of the coordinate system; thus translating or reflections of the coordinate
  system result in the same boxes being selected by the algorithm.
  The output of this operation is a set of integers indexing into the input
  collection of bounding boxes representing the selected boxes.  The bounding
  box coordinates corresponding to the selected indices can then be obtained
  using the `tf.gather` operation.  For example:
    ```python
    selected_indices, selected_scores = tf.image.non_max_suppression_padded(
        boxes, scores, max_output_size, iou_threshold=1.0, score_threshold=0.1,
        soft_nms_sigma=0.5)
    selected_boxes = tf.gather(boxes, selected_indices)
    ```
  This function generalizes the `tf.image.non_max_suppression` op by also
  supporting a Soft-NMS (with Gaussian weighting) mode (c.f.
  Bodla et al, https://arxiv.org/abs/1704.04503) where boxes reduce the score
  of other overlapping boxes instead of directly causing them to be pruned.
  Consequently, in contrast to `tf.image.non_max_suppression`,
  `tf.image.non_max_suppression_with_scores` returns the new scores of each
  input box in the second output, `selected_scores`.
  To enable this Soft-NMS mode, set the `soft_nms_sigma` parameter to be
  larger than 0.  When `soft_nms_sigma` equals 0, the behavior of
  `tf.image.non_max_suppression_with_scores` is identical to that of
  `tf.image.non_max_suppression` (except for the extra output) both in function
  and in running time.
  Note that when `soft_nms_sigma` > 0, Soft-NMS is performed and `iou_threshold`
  is ignored. `iou_threshold` is only used for standard NMS.
  Args:
    boxes: A 2-D float `Tensor` of shape `[num_boxes, 4]`.
    scores: A 1-D float `Tensor` of shape `[num_boxes]` representing a single
      score corresponding to each box (each row of boxes).
    max_output_size: A scalar integer `Tensor` representing the maximum number
      of boxes to be selected by non-max suppression.
    iou_threshold: A 0-D float tensor representing the threshold for deciding
      whether boxes overlap too much with respect to IOU.
    score_threshold: A 0-D float tensor representing the threshold for deciding
      when to remove boxes based on score.
    soft_nms_sigma: A 0-D float tensor representing the sigma parameter for Soft
      NMS; see Bodla et al (c.f. https://arxiv.org/abs/1704.04503).  When
      `soft_nms_sigma=0.0` (which is default), we fall back to standard (hard)
      NMS.
    name: A name for the operation (optional).
  Returns:
    selected_indices: A 1-D integer `Tensor` of shape `[M]` representing the
      selected indices from the boxes tensor, where `M <= max_output_size`.
    selected_scores: A 1-D float tensor of shape `[M]` representing the
      corresponding scores for each selected box, where `M <= max_output_size`.
      Scores only differ from corresponding input scores when using Soft NMS
      (i.e. when `soft_nms_sigma>0`)
  """
  with ops.name_scope(name, 'non_max_suppression_with_scores'):
    iou_threshold = ops.convert_to_tensor(iou_threshold, name='iou_threshold')
    score_threshold = ops.convert_to_tensor(
        score_threshold, name='score_threshold')
    soft_nms_sigma = ops.convert_to_tensor(
        soft_nms_sigma, name='soft_nms_sigma')
    (selected_indices, selected_scores,
     _) = gen_image_ops.non_max_suppression_v5(
         boxes,
         scores,
         max_output_size,
         iou_threshold,
         score_threshold,
         soft_nms_sigma,
         pad_to_max_output_size=False)
    return selected_indices, selected_scores
@tf_export('image.non_max_suppression_overlaps')
@dispatch.add_dispatch_support
def non_max_suppression_with_overlaps(overlaps,
                                      scores,
                                      max_output_size,
                                      overlap_threshold=0.5,
                                      score_threshold=float('-inf'),
                                      name=None):
  """Greedily selects a subset of bounding boxes in descending order of score.
  Prunes away boxes that have high overlap with previously selected boxes.
  N-by-n overlap values are supplied as square matrix.
  The output of this operation is a set of integers indexing into the input
  collection of bounding boxes representing the selected boxes.  The bounding
  box coordinates corresponding to the selected indices can then be obtained
  using the `tf.gather` operation.  For example:
    ```python
    selected_indices = tf.image.non_max_suppression_overlaps(
        overlaps, scores, max_output_size, iou_threshold)
    selected_boxes = tf.gather(boxes, selected_indices)
    ```
  Args:
    overlaps: A 2-D float `Tensor` of shape `[num_boxes, num_boxes]`
      representing the n-by-n box overlap values.
    scores: A 1-D float `Tensor` of shape `[num_boxes]` representing a single
      score corresponding to each box (each row of boxes).
    max_output_size: A scalar integer `Tensor` representing the maximum number
      of boxes to be selected by non-max suppression.
    overlap_threshold: A 0-D float tensor representing the threshold for
      deciding whether boxes overlap too much with respect to the provided
      overlap values.
    score_threshold: A 0-D float tensor representing the threshold for deciding
      when to remove boxes based on score.
    name: A name for the operation (optional).
  Returns:
    selected_indices: A 1-D integer `Tensor` of shape `[M]` representing the
      selected indices from the overlaps tensor, where `M <= max_output_size`.
  """
  with ops.name_scope(name, 'non_max_suppression_overlaps'):
    overlap_threshold = ops.convert_to_tensor(
        overlap_threshold, name='overlap_threshold')
    # pylint: disable=protected-access
    return gen_image_ops.non_max_suppression_with_overlaps(
        overlaps, scores, max_output_size, overlap_threshold, score_threshold)
    # pylint: enable=protected-access
_rgb_to_yiq_kernel = [[0.299, 0.59590059, 0.2115],
                      [0.587, -0.27455667, -0.52273617],
                      [0.114, -0.32134392, 0.31119955]]
@tf_export('image.rgb_to_yiq')
@dispatch.add_dispatch_support
def rgb_to_yiq(images):
  """Converts one or more images from RGB to YIQ.
  Outputs a tensor of the same shape as the `images` tensor, containing the YIQ
  value of the pixels.
  The output is only well defined if the value in images are in [0,1].
  Usage Example:
  >>> x = tf.constant([[[1.0, 2.0, 3.0]]])
  >>> tf.image.rgb_to_yiq(x)
  <tf.Tensor: shape=(1, 1, 3), dtype=float32,
  numpy=array([[[ 1.815     , -0.91724455,  0.09962624]]], dtype=float32)>
  Args:
    images: 2-D or higher rank. Image data to convert. Last dimension must be
      size 3.
  Returns:
    images: tensor with the same shape as `images`.
  """
  images = ops.convert_to_tensor(images, name='images')
  kernel = ops.convert_to_tensor(
      _rgb_to_yiq_kernel, dtype=images.dtype, name='kernel')
  ndims = images.get_shape().ndims
  return math_ops.tensordot(images, kernel, axes=[[ndims - 1], [0]])
_yiq_to_rgb_kernel = [[1, 1, 1], [0.95598634, -0.27201283, -1.10674021],
                      [0.6208248, -0.64720424, 1.70423049]]
@tf_export('image.yiq_to_rgb')
@dispatch.add_dispatch_support
def yiq_to_rgb(images):
  """Converts one or more images from YIQ to RGB.
  Outputs a tensor of the same shape as the `images` tensor, containing the RGB
  value of the pixels.
  The output is only well defined if the Y value in images are in [0,1],
  I value are in [-0.5957,0.5957] and Q value are in [-0.5226,0.5226].
  Args:
    images: 2-D or higher rank. Image data to convert. Last dimension must be
      size 3.
  Returns:
    images: tensor with the same shape as `images`.
  """
  images = ops.convert_to_tensor(images, name='images')
  kernel = ops.convert_to_tensor(
      _yiq_to_rgb_kernel, dtype=images.dtype, name='kernel')
  ndims = images.get_shape().ndims
  return math_ops.tensordot(images, kernel, axes=[[ndims - 1], [0]])
_rgb_to_yuv_kernel = [[0.299, -0.14714119, 0.61497538],
                      [0.587, -0.28886916, -0.51496512],
                      [0.114, 0.43601035, -0.10001026]]
@tf_export('image.rgb_to_yuv')
@dispatch.add_dispatch_support
def rgb_to_yuv(images):
  """Converts one or more images from RGB to YUV.
  Outputs a tensor of the same shape as the `images` tensor, containing the YUV
  value of the pixels.
  The output is only well defined if the value in images are in [0, 1].
  There are two ways of representing an image: [0, 255] pixel values range or
  [0, 1] (as float) pixel values range. Users need to convert the input image
  into a float [0, 1] range.
  Args:
    images: 2-D or higher rank. Image data to convert. Last dimension must be
      size 3.
  Returns:
    images: tensor with the same shape as `images`.
  """
  images = ops.convert_to_tensor(images, name='images')
  kernel = ops.convert_to_tensor(
      _rgb_to_yuv_kernel, dtype=images.dtype, name='kernel')
  ndims = images.get_shape().ndims
  return math_ops.tensordot(images, kernel, axes=[[ndims - 1], [0]])
_yuv_to_rgb_kernel = [[1, 1, 1], [0, -0.394642334, 2.03206185],
                      [1.13988303, -0.58062185, 0]]
@tf_export('image.yuv_to_rgb')
@dispatch.add_dispatch_support
def yuv_to_rgb(images):
  """Converts one or more images from YUV to RGB.
  Outputs a tensor of the same shape as the `images` tensor, containing the RGB
  value of the pixels.
  The output is only well defined if the Y value in images are in [0,1],
  U and V value are in [-0.5,0.5].
  As per the above description, you need to scale your YUV images if their
  pixel values are not in the required range. Below given example illustrates
  preprocessing of each channel of images before feeding them to `yuv_to_rgb`.
  ```python
  yuv_images = tf.random.uniform(shape=[100, 64, 64, 3], maxval=255)
  last_dimension_axis = len(yuv_images.shape) - 1
  yuv_tensor_images = tf.truediv(
      tf.subtract(
          yuv_images,
          tf.reduce_min(yuv_images)
      ),
      tf.subtract(
          tf.reduce_max(yuv_images),
          tf.reduce_min(yuv_images)
       )
  )
  y, u, v = tf.split(yuv_tensor_images, 3, axis=last_dimension_axis)
  target_uv_min, target_uv_max = -0.5, 0.5
  u = u * (target_uv_max - target_uv_min) + target_uv_min
  v = v * (target_uv_max - target_uv_min) + target_uv_min
  preprocessed_yuv_images = tf.concat([y, u, v], axis=last_dimension_axis)
  rgb_tensor_images = tf.image.yuv_to_rgb(preprocessed_yuv_images)
  ```
  Args:
    images: 2-D or higher rank. Image data to convert. Last dimension must be
      size 3.
  Returns:
    images: tensor with the same shape as `images`.
  """
  images = ops.convert_to_tensor(images, name='images')
  kernel = ops.convert_to_tensor(
      _yuv_to_rgb_kernel, dtype=images.dtype, name='kernel')
  ndims = images.get_shape().ndims
  return math_ops.tensordot(images, kernel, axes=[[ndims - 1], [0]])
def _verify_compatible_image_shapes(img1, img2):
  """Checks if two image tensors are compatible for applying SSIM or PSNR.
  This function checks if two sets of images have ranks at least 3, and if the
  last three dimensions match.
  Args:
    img1: Tensor containing the first image batch.
    img2: Tensor containing the second image batch.
  Returns:
    A tuple containing: the first tensor shape, the second tensor shape, and a
    list of control_flow_ops.Assert() ops implementing the checks.
  Raises:
    ValueError: When static shape check fails.
  """
  shape1 = img1.get_shape().with_rank_at_least(3)
  shape2 = img2.get_shape().with_rank_at_least(3)
  shape1[-3:].assert_is_compatible_with(shape2[-3:])
  if shape1.ndims is not None and shape2.ndims is not None:
    for dim1, dim2 in zip(
        reversed(shape1.dims[:-3]), reversed(shape2.dims[:-3])):
      if not (dim1 == 1 or dim2 == 1 or dim1.is_compatible_with(dim2)):
        raise ValueError('Two images are not compatible: %s and %s' %
                         (shape1, shape2))
  # Now assign shape tensors.
  shape1, shape2 = array_ops.shape_n([img1, img2])
  # TODO(sjhwang): Check if shape1[:-3] and shape2[:-3] are broadcastable.
  checks = []
  checks.append(
      control_flow_assert.Assert(
          math_ops.greater_equal(array_ops.size(shape1), 3), [shape1, shape2],
          summarize=10))
  checks.append(
      control_flow_assert.Assert(
          math_ops.reduce_all(math_ops.equal(shape1[-3:], shape2[-3:])),
          [shape1, shape2],
          summarize=10))
  return shape1, shape2, checks
@tf_export('image.psnr')
@dispatch.add_dispatch_support
def psnr(a, b, max_val, name=None):
  """Returns the Peak Signal-to-Noise Ratio between a and b.
  This is intended to be used on signals (or images). Produces a PSNR value for
  each image in batch.
  The last three dimensions of input are expected to be [height, width, depth].
  Example:
  ```python
      # Read images from file.
      im1 = tf.decode_png('path/to/im1.png')
      im2 = tf.decode_png('path/to/im2.png')
      # Compute PSNR over tf.uint8 Tensors.
      psnr1 = tf.image.psnr(im1, im2, max_val=255)
      # Compute PSNR over tf.float32 Tensors.
      im1 = tf.image.convert_image_dtype(im1, tf.float32)
      im2 = tf.image.convert_image_dtype(im2, tf.float32)
      psnr2 = tf.image.psnr(im1, im2, max_val=1.0)
      # psnr1 and psnr2 both have type tf.float32 and are almost equal.
  ```
  Args:
    a: First set of images.
    b: Second set of images.
    max_val: The dynamic range of the images (i.e., the difference between the
      maximum the and minimum allowed values).
    name: Namespace to embed the computation in.
  Returns:
    The scalar PSNR between a and b. The returned tensor has type `tf.float32`
    and shape [batch_size, 1].
  """
  with ops.name_scope(name, 'PSNR', [a, b]):
    # Need to convert the images to float32.  Scale max_val accordingly so that
    # PSNR is computed correctly.
    max_val = math_ops.cast(max_val, a.dtype)
    max_val = convert_image_dtype(max_val, dtypes.float32)
    a = convert_image_dtype(a, dtypes.float32)
    b = convert_image_dtype(b, dtypes.float32)
    mse = math_ops.reduce_mean(math_ops.squared_difference(a, b), [-3, -2, -1])
    psnr_val = math_ops.subtract(
        20 * math_ops.log(max_val) / math_ops.log(10.0),
        np.float32(10 / np.log(10)) * math_ops.log(mse),
        name='psnr')
    _, _, checks = _verify_compatible_image_shapes(a, b)
    with ops.control_dependencies(checks):
      return array_ops.identity(psnr_val)
def _ssim_helper(x, y, reducer, max_val, compensation=1.0, k1=0.01, k2=0.03):
  r"""Helper function for computing SSIM.
  SSIM estimates covariances with weighted sums.  The default parameters
  use a biased estimate of the covariance:
  Suppose `reducer` is a weighted sum, then the mean estimators are
    \mu_x = \sum_i w_i x_i,
    \mu_y = \sum_i w_i y_i,
  where w_i's are the weighted-sum weights, and covariance estimator is
    cov_{xy} = \sum_i w_i (x_i - \mu_x) (y_i - \mu_y)
  with assumption \sum_i w_i = 1. This covariance estimator is biased, since
    E[cov_{xy}] = (1 - \sum_i w_i ^ 2) Cov(X, Y).
  For SSIM measure with unbiased covariance estimators, pass as `compensation`
  argument (1 - \sum_i w_i ^ 2).
  Args:
    x: First set of images.
    y: Second set of images.
    reducer: Function that computes 'local' averages from the set of images. For
      non-convolutional version, this is usually tf.reduce_mean(x, [1, 2]), and
      for convolutional version, this is usually tf.nn.avg_pool2d or
      tf.nn.conv2d with weighted-sum kernel.
    max_val: The dynamic range (i.e., the difference between the maximum
      possible allowed value and the minimum allowed value).
    compensation: Compensation factor. See above.
    k1: Default value 0.01
    k2: Default value 0.03 (SSIM is less sensitivity to K2 for lower values, so
      it would be better if we took the values in the range of 0 < K2 < 0.4).
  Returns:
    A pair containing the luminance measure, and the contrast-structure measure.
  """
  c1 = (k1 * max_val)**2
  c2 = (k2 * max_val)**2
  # SSIM luminance measure is
  # (2 * mu_x * mu_y + c1) / (mu_x ** 2 + mu_y ** 2 + c1).
  mean0 = reducer(x)
  mean1 = reducer(y)
  num0 = mean0 * mean1 * 2.0
  den0 = math_ops.square(mean0) + math_ops.square(mean1)
  luminance = (num0 + c1) / (den0 + c1)
  # SSIM contrast-structure measure is
  #   (2 * cov_{xy} + c2) / (cov_{xx} + cov_{yy} + c2).
  # Note that `reducer` is a weighted sum with weight w_k, \sum_i w_i = 1, then
  #   cov_{xy} = \sum_i w_i (x_i - \mu_x) (y_i - \mu_y)
  #          = \sum_i w_i x_i y_i - (\sum_i w_i x_i) (\sum_j w_j y_j).
  num1 = reducer(x * y) * 2.0
  den1 = reducer(math_ops.square(x) + math_ops.square(y))
  c2 *= compensation
  cs = (num1 - num0 + c2) / (den1 - den0 + c2)
  # SSIM score is the product of the luminance and contrast-structure measures.
  return luminance, cs
def _fspecial_gauss(size, sigma):
  """Function to mimic the 'fspecial' gaussian MATLAB function."""
  size = ops.convert_to_tensor(size, dtypes.int32)
  sigma = ops.convert_to_tensor(sigma)
  coords = math_ops.cast(math_ops.range(size), sigma.dtype)
  coords -= math_ops.cast(size - 1, sigma.dtype) / 2.0
  g = math_ops.square(coords)
  g *= -0.5 / math_ops.square(sigma)
  g = array_ops.reshape(g, shape=[1, -1]) + array_ops.reshape(g, shape=[-1, 1])
  g = array_ops.reshape(g, shape=[1, -1])  # For tf.nn.softmax().
  g = nn_ops.softmax(g)
  return array_ops.reshape(g, shape=[size, size, 1, 1])
def _ssim_per_channel(img1,
                      img2,
                      max_val=1.0,
                      filter_size=11,
                      filter_sigma=1.5,
                      k1=0.01,
                      k2=0.03,
                      return_index_map=False):
  """Computes SSIM index between img1 and img2 per color channel.
  This function matches the standard SSIM implementation from:
  Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image
  quality assessment: from error visibility to structural similarity. IEEE
  transactions on image processing.
  Details:
    - 11x11 Gaussian filter of width 1.5 is used.
    - k1 = 0.01, k2 = 0.03 as in the original paper.
  Args:
    img1: First image batch.
    img2: Second image batch.
    max_val: The dynamic range of the images (i.e., the difference between the
      maximum the and minimum allowed values).
    filter_size: Default value 11 (size of gaussian filter).
    filter_sigma: Default value 1.5 (width of gaussian filter).
    k1: Default value 0.01
    k2: Default value 0.03 (SSIM is less sensitivity to K2 for lower values, so
      it would be better if we took the values in the range of 0 < K2 < 0.4).
    return_index_map: If True returns local SSIM map instead of the global mean.
  Returns:
    A pair of tensors containing and channel-wise SSIM and contrast-structure
    values. The shape is [..., channels].
  """
  filter_size = constant_op.constant(filter_size, dtype=dtypes.int32)
  filter_sigma = constant_op.constant(filter_sigma, dtype=img1.dtype)
  shape1, shape2 = array_ops.shape_n([img1, img2])
  checks = [
      control_flow_assert.Assert(
          math_ops.reduce_all(
              math_ops.greater_equal(shape1[-3:-1], filter_size)),
          [shape1, filter_size],
          summarize=8),
      control_flow_assert.Assert(
          math_ops.reduce_all(
              math_ops.greater_equal(shape2[-3:-1], filter_size)),
          [shape2, filter_size],
          summarize=8)
  ]
  # Enforce the check to run before computation.
  with ops.control_dependencies(checks):
    img1 = array_ops.identity(img1)
  # TODO(sjhwang): Try to cache kernels and compensation factor.
  kernel = _fspecial_gauss(filter_size, filter_sigma)
  kernel = array_ops.tile(kernel, multiples=[1, 1, shape1[-1], 1])
  # The correct compensation factor is `1.0 - tf.reduce_sum(tf.square(kernel))`,
  # but to match MATLAB implementation of MS-SSIM, we use 1.0 instead.
  compensation = 1.0
  # TODO(sjhwang): Try FFT.
  # TODO(sjhwang): Gaussian kernel is separable in space. Consider applying
  #   1-by-n and n-by-1 Gaussian filters instead of an n-by-n filter.
  def reducer(x):
    shape = array_ops.shape(x)
    x = array_ops.reshape(x, shape=array_ops.concat([[-1], shape[-3:]], 0))
    y = nn.depthwise_conv2d(x, kernel, strides=[1, 1, 1, 1], padding='VALID')
    return array_ops.reshape(
        y, array_ops.concat([shape[:-3], array_ops.shape(y)[1:]], 0))
  luminance, cs = _ssim_helper(img1, img2, reducer, max_val, compensation, k1,
                               k2)
  # Average over the second and the third from the last: height, width.
  if return_index_map:
    ssim_val = luminance * cs
  else:
    axes = constant_op.constant([-3, -2], dtype=dtypes.int32)
    ssim_val = math_ops.reduce_mean(luminance * cs, axes)
    cs = math_ops.reduce_mean(cs, axes)
  return ssim_val, cs
@tf_export('image.ssim')
@dispatch.add_dispatch_support
def ssim(img1,
         img2,
         max_val,
         filter_size=11,
         filter_sigma=1.5,
         k1=0.01,
         k2=0.03,
         return_index_map=False):
  """Computes SSIM index between img1 and img2.
  This function is based on the standard SSIM implementation from:
  Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image
  quality assessment: from error visibility to structural similarity. IEEE
  transactions on image processing.
  Note: The true SSIM is only defined on grayscale.  This function does not
  perform any colorspace transform.  (If the input is already YUV, then it will
  compute YUV SSIM average.)
  Details:
    - 11x11 Gaussian filter of width 1.5 is used.
    - k1 = 0.01, k2 = 0.03 as in the original paper.
  The image sizes must be at least 11x11 because of the filter size.
  Example:
  ```python
      # Read images (of size 255 x 255) from file.
      im1 = tf.image.decode_image(tf.io.read_file('path/to/im1.png'))
      im2 = tf.image.decode_image(tf.io.read_file('path/to/im2.png'))
      tf.shape(im1)  # `img1.png` has 3 channels; shape is `(255, 255, 3)`
      tf.shape(im2)  # `img2.png` has 3 channels; shape is `(255, 255, 3)`
      # Add an outer batch for each image.
      im1 = tf.expand_dims(im1, axis=0)
      im2 = tf.expand_dims(im2, axis=0)
      # Compute SSIM over tf.uint8 Tensors.
      ssim1 = tf.image.ssim(im1, im2, max_val=255, filter_size=11,
                            filter_sigma=1.5, k1=0.01, k2=0.03)
      # Compute SSIM over tf.float32 Tensors.
      im1 = tf.image.convert_image_dtype(im1, tf.float32)
      im2 = tf.image.convert_image_dtype(im2, tf.float32)
      ssim2 = tf.image.ssim(im1, im2, max_val=1.0, filter_size=11,
                            filter_sigma=1.5, k1=0.01, k2=0.03)
      # ssim1 and ssim2 both have type tf.float32 and are almost equal.
  ```
  Args:
    img1: First image batch. 4-D Tensor of shape `[batch, height, width,
      channels]` with only Positive Pixel Values.
    img2: Second image batch. 4-D Tensor of shape `[batch, height, width,
      channels]` with only Positive Pixel Values.
    max_val: The dynamic range of the images (i.e., the difference between the
      maximum the and minimum allowed values).
    filter_size: Default value 11 (size of gaussian filter).
    filter_sigma: Default value 1.5 (width of gaussian filter).
    k1: Default value 0.01
    k2: Default value 0.03 (SSIM is less sensitivity to K2 for lower values, so
      it would be better if we took the values in the range of 0 < K2 < 0.4).
    return_index_map: If True returns local SSIM map instead of the global mean.
  Returns:
    A tensor containing an SSIM value for each image in batch or a tensor
    containing an SSIM value for each pixel for each image in batch if
    return_index_map is True. Returned SSIM values are in range (-1, 1], when
    pixel values are non-negative. Returns a tensor with shape:
    broadcast(img1.shape[:-3], img2.shape[:-3]) or broadcast(img1.shape[:-1],
    img2.shape[:-1]).
  """
  with ops.name_scope(None, 'SSIM', [img1, img2]):
    # Convert to tensor if needed.
    img1 = ops.convert_to_tensor(img1, name='img1')
    img2 = ops.convert_to_tensor(img2, name='img2')
    # Shape checking.
    _, _, checks = _verify_compatible_image_shapes(img1, img2)
    with ops.control_dependencies(checks):
      img1 = array_ops.identity(img1)
    # Need to convert the images to float32.  Scale max_val accordingly so that
    # SSIM is computed correctly.
    max_val = math_ops.cast(max_val, img1.dtype)
    max_val = convert_image_dtype(max_val, dtypes.float32)
    img1 = convert_image_dtype(img1, dtypes.float32)
    img2 = convert_image_dtype(img2, dtypes.float32)
    ssim_per_channel, _ = _ssim_per_channel(img1, img2, max_val, filter_size,
                                            filter_sigma, k1, k2,
                                            return_index_map)
    # Compute average over color channels.
    return math_ops.reduce_mean(ssim_per_channel, [-1])
# Default values obtained by Wang et al.
_MSSSIM_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)
@tf_export('image.ssim_multiscale')
@dispatch.add_dispatch_support
def ssim_multiscale(img1,
                    img2,
                    max_val,
                    power_factors=_MSSSIM_WEIGHTS,
                    filter_size=11,
                    filter_sigma=1.5,
                    k1=0.01,
                    k2=0.03):
  """Computes the MS-SSIM between img1 and img2.
  This function assumes that `img1` and `img2` are image batches, i.e. the last
  three dimensions are [height, width, channels].
  Note: The true SSIM is only defined on grayscale.  This function does not
  perform any colorspace transform.  (If the input is already YUV, then it will
  compute YUV SSIM average.)
  Original paper: Wang, Zhou, Eero P. Simoncelli, and Alan C. Bovik. "Multiscale
  structural similarity for image quality assessment." Signals, Systems and
  Computers, 2004.
  Args:
    img1: First image batch with only Positive Pixel Values.
    img2: Second image batch with only Positive Pixel Values. Must have the
    same rank as img1.
    max_val: The dynamic range of the images (i.e., the difference between the
      maximum the and minimum allowed values).
    power_factors: Iterable of weights for each of the scales. The number of
      scales used is the length of the list. Index 0 is the unscaled
      resolution's weight and each increasing scale corresponds to the image
      being downsampled by 2.  Defaults to (0.0448, 0.2856, 0.3001, 0.2363,
      0.1333), which are the values obtained in the original paper.
    filter_size: Default value 11 (size of gaussian filter).
    filter_sigma: Default value 1.5 (width of gaussian filter).
    k1: Default value 0.01
    k2: Default value 0.03 (SSIM is less sensitivity to K2 for lower values, so
      it would be better if we took the values in the range of 0 < K2 < 0.4).
  Returns:
    A tensor containing an MS-SSIM value for each image in batch.  The values
    are in range [0, 1].  Returns a tensor with shape:
    broadcast(img1.shape[:-3], img2.shape[:-3]).
  """
  with ops.name_scope(None, 'MS-SSIM', [img1, img2]):
    # Convert to tensor if needed.
    img1 = ops.convert_to_tensor(img1, name='img1')
    img2 = ops.convert_to_tensor(img2, name='img2')
    # Shape checking.
    shape1, shape2, checks = _verify_compatible_image_shapes(img1, img2)
    with ops.control_dependencies(checks):
      img1 = array_ops.identity(img1)
    # Need to convert the images to float32.  Scale max_val accordingly so that
    # SSIM is computed correctly.
    max_val = math_ops.cast(max_val, img1.dtype)
    max_val = convert_image_dtype(max_val, dtypes.float32)
    img1 = convert_image_dtype(img1, dtypes.float32)
    img2 = convert_image_dtype(img2, dtypes.float32)
    imgs = [img1, img2]
    shapes = [shape1, shape2]
    # img1 and img2 are assumed to be a (multi-dimensional) batch of
    # 3-dimensional images (height, width, channels). `heads` contain the batch
    # dimensions, and `tails` contain the image dimensions.
    heads = [s[:-3] for s in shapes]
    tails = [s[-3:] for s in shapes]
    divisor = [1, 2, 2, 1]
    divisor_tensor = constant_op.constant(divisor[1:], dtype=dtypes.int32)
    def do_pad(images, remainder):
      padding = array_ops.expand_dims(remainder, -1)
      padding = array_ops.pad(padding, [[1, 0], [1, 0]])
      return [array_ops.pad(x, padding, mode='SYMMETRIC') for x in images]
    mcs = []
    for k in range(len(power_factors)):
      with ops.name_scope(None, 'Scale%d' % k, imgs):
        if k > 0:
          # Avg pool takes rank 4 tensors. Flatten leading dimensions.
          flat_imgs = [
              array_ops.reshape(x, array_ops.concat([[-1], t], 0))
              for x, t in zip(imgs, tails)
          ]
          remainder = tails[0] % divisor_tensor
          need_padding = math_ops.reduce_any(math_ops.not_equal(remainder, 0))
          # pylint: disable=cell-var-from-loop
          padded = tf_cond.cond(need_padding,
                                lambda: do_pad(flat_imgs, remainder),
                                lambda: flat_imgs)
          # pylint: enable=cell-var-from-loop
          downscaled = [
              nn_ops.avg_pool(
                  x, ksize=divisor, strides=divisor, padding='VALID')
              for x in padded
          ]
          tails = [x[1:] for x in array_ops.shape_n(downscaled)]
          imgs = [
              array_ops.reshape(x, array_ops.concat([h, t], 0))
              for x, h, t in zip(downscaled, heads, tails)
          ]
        # Overwrite previous ssim value since we only need the last one.
        ssim_per_channel, cs = _ssim_per_channel(
            *imgs,
            max_val=max_val,
            filter_size=filter_size,
            filter_sigma=filter_sigma,
            k1=k1,
            k2=k2)
        mcs.append(nn_ops.relu(cs))
    # Remove the cs score for the last scale. In the MS-SSIM calculation,
    # we use the l(p) at the highest scale. l(p) * cs(p) is ssim(p).
    mcs.pop()  # Remove the cs score for the last scale.
    mcs_and_ssim = array_ops_stack.stack(
        mcs + [nn_ops.relu(ssim_per_channel)], axis=-1)
    # Take weighted geometric mean across the scale axis.
    ms_ssim = math_ops.reduce_prod(
        math_ops.pow(mcs_and_ssim, power_factors), [-1])
    return math_ops.reduce_mean(ms_ssim, [-1])  # Avg over color channels.
@tf_export('image.image_gradients')
@dispatch.add_dispatch_support
def image_gradients(image):
  """Returns image gradients (dy, dx) for each color channel.
  Both output tensors have the same shape as the input: [batch_size, h, w,
  d]. The gradient values are organized so that [I(x+1, y) - I(x, y)] is in
  location (x, y). That means that dy will always have zeros in the last row,
  and dx will always have zeros in the last column.
  Usage Example:
    ```python
    BATCH_SIZE = 1
    IMAGE_HEIGHT = 5
    IMAGE_WIDTH = 5
    CHANNELS = 1
    image = tf.reshape(tf.range(IMAGE_HEIGHT * IMAGE_WIDTH * CHANNELS,
      delta=1, dtype=tf.float32),
      shape=(BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
    dy, dx = tf.image.image_gradients(image)
    print(image[0, :,:,0])
    tf.Tensor(
      [[ 0.  1.  2.  3.  4.]
      [ 5.  6.  7.  8.  9.]
      [10. 11. 12. 13. 14.]
      [15. 16. 17. 18. 19.]
      [20. 21. 22. 23. 24.]], shape=(5, 5), dtype=float32)
    print(dy[0, :,:,0])
    tf.Tensor(
      [[5. 5. 5. 5. 5.]
      [5. 5. 5. 5. 5.]
      [5. 5. 5. 5. 5.]
      [5. 5. 5. 5. 5.]
      [0. 0. 0. 0. 0.]], shape=(5, 5), dtype=float32)
    print(dx[0, :,:,0])
    tf.Tensor(
      [[1. 1. 1. 1. 0.]
      [1. 1. 1. 1. 0.]
      [1. 1. 1. 1. 0.]
      [1. 1. 1. 1. 0.]
      [1. 1. 1. 1. 0.]], shape=(5, 5), dtype=float32)
    ```
  Args:
    image: Tensor with shape [batch_size, h, w, d].
  Returns:
    Pair of tensors (dy, dx) holding the vertical and horizontal image
    gradients (1-step finite difference).
  Raises:
    ValueError: If `image` is not a 4D tensor.
  """
  if image.get_shape().ndims != 4:
    raise ValueError('image_gradients expects a 4D tensor '
                     '[batch_size, h, w, d], not {}.'.format(image.get_shape()))
  image_shape = array_ops.shape(image)
  batch_size, height, width, depth = array_ops_stack.unstack(image_shape)
  dy = image[:, 1:, :, :] - image[:, :-1, :, :]
  dx = image[:, :, 1:, :] - image[:, :, :-1, :]
  # Return tensors with same size as original image by concatenating
  # zeros. Place the gradient [I(x+1,y) - I(x,y)] on the base pixel (x, y).
  shape = array_ops_stack.stack([batch_size, 1, width, depth])
  dy = array_ops.concat([dy, array_ops.zeros(shape, image.dtype)], 1)
  dy = array_ops.reshape(dy, image_shape)
  shape = array_ops_stack.stack([batch_size, height, 1, depth])
  dx = array_ops.concat([dx, array_ops.zeros(shape, image.dtype)], 2)
  dx = array_ops.reshape(dx, image_shape)
  return dy, dx
@tf_export('image.sobel_edges')
@dispatch.add_dispatch_support
def sobel_edges(image):
  """Returns a tensor holding Sobel edge maps.
  Example usage:
  For general usage, `image` would be loaded from a file as below:
  ```python
  image_bytes = tf.io.read_file(path_to_image_file)
  image = tf.image.decode_image(image_bytes)
  image = tf.cast(image, tf.float32)
  image = tf.expand_dims(image, 0)
  ```
  But for demo purposes, we are using randomly generated values for `image`:
  >>> image = tf.random.uniform(
  ...   maxval=255, shape=[1, 28, 28, 3], dtype=tf.float32)
  >>> sobel = tf.image.sobel_edges(image)
  >>> sobel_y = np.asarray(sobel[0, :, :, :, 0]) # sobel in y-direction
  >>> sobel_x = np.asarray(sobel[0, :, :, :, 1]) # sobel in x-direction
  For displaying the sobel results, PIL's [Image Module](
  https://pillow.readthedocs.io/en/stable/reference/Image.html) can be used:
  ```python
  # Display edge maps for the first channel (at index 0)
  Image.fromarray(sobel_y[..., 0] / 4 + 0.5).show()
  Image.fromarray(sobel_x[..., 0] / 4 + 0.5).show()
  ```
  Args:
    image: Image tensor with shape [batch_size, h, w, d] and type float32 or
      float64.  The image(s) must be 2x2 or larger.
  Returns:
    Tensor holding edge maps for each channel. Returns a tensor with shape
    [batch_size, h, w, d, 2] where the last two dimensions hold [[dy[0], dx[0]],
    [dy[1], dx[1]], ..., [dy[d-1], dx[d-1]]] calculated using the Sobel filter.
  """
  # Define vertical and horizontal Sobel filters.
  static_image_shape = image.get_shape()
  image_shape = array_ops.shape(image)
  kernels = [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
             [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]
  num_kernels = len(kernels)
  kernels = np.transpose(np.asarray(kernels), (1, 2, 0))
  kernels = np.expand_dims(kernels, -2)
  kernels_tf = constant_op.constant(kernels, dtype=image.dtype)
  kernels_tf = array_ops.tile(
      kernels_tf, [1, 1, image_shape[-1], 1], name='sobel_filters')
  # Use depth-wise convolution to calculate edge maps per channel.
  pad_sizes = [[0, 0], [1, 1], [1, 1], [0, 0]]
  padded = array_ops.pad(image, pad_sizes, mode='REFLECT')
  # Output tensor has shape [batch_size, h, w, d * num_kernels].
  strides = [1, 1, 1, 1]
  output = nn.depthwise_conv2d(padded, kernels_tf, strides, 'VALID')
  # Reshape to [batch_size, h, w, d, num_kernels].
  shape = array_ops.concat([image_shape, [num_kernels]], 0)
  output = array_ops.reshape(output, shape=shape)
  output.set_shape(static_image_shape.concatenate([num_kernels]))
  return output
def resize_bicubic(images,
                   size,
                   align_corners=False,
                   name=None,
                   half_pixel_centers=False):
  return gen_image_ops.resize_bicubic(
      images=images,
      size=size,
      align_corners=align_corners,
      half_pixel_centers=half_pixel_centers,
      name=name)
def resize_bilinear(images,
                    size,
                    align_corners=False,
                    name=None,
                    half_pixel_centers=False):
  return gen_image_ops.resize_bilinear(
      images=images,
      size=size,
      align_corners=align_corners,
      half_pixel_centers=half_pixel_centers,
      name=name)
def resize_nearest_neighbor(images,
                            size,
                            align_corners=False,
                            name=None,
                            half_pixel_centers=False):
  return gen_image_ops.resize_nearest_neighbor(
      images=images,
      size=size,
      align_corners=align_corners,
      half_pixel_centers=half_pixel_centers,
      name=name)
resize_area_deprecation = deprecation.deprecated(
    date=None,
    instructions=(
        'Use `tf.image.resize(...method=ResizeMethod.AREA...)` instead.'))
tf_export(v1=['image.resize_area'])(
    resize_area_deprecation(
        dispatch.add_dispatch_support(gen_image_ops.resize_area)))
resize_bicubic_deprecation = deprecation.deprecated(
    date=None,
    instructions=(
        'Use `tf.image.resize(...method=ResizeMethod.BICUBIC...)` instead.'))
tf_export(v1=['image.resize_bicubic'])(
    dispatch.add_dispatch_support(resize_bicubic_deprecation(resize_bicubic)))
resize_bilinear_deprecation = deprecation.deprecated(
    date=None,
    instructions=(
        'Use `tf.image.resize(...method=ResizeMethod.BILINEAR...)` instead.'))
tf_export(v1=['image.resize_bilinear'])(
    dispatch.add_dispatch_support(resize_bilinear_deprecation(resize_bilinear)))
resize_nearest_neighbor_deprecation = deprecation.deprecated(
    date=None,
    instructions=(
        'Use `tf.image.resize(...method=ResizeMethod.NEAREST_NEIGHBOR...)` '
        'instead.'))
tf_export(v1=['image.resize_nearest_neighbor'])(
    dispatch.add_dispatch_support(
        resize_nearest_neighbor_deprecation(resize_nearest_neighbor)))
@tf_export('image.crop_and_resize', v1=[])
@dispatch.add_dispatch_support
def crop_and_resize_v2(image,
                       boxes,
                       box_indices,
                       crop_size,
                       method='bilinear',
                       extrapolation_value=.0,
                       name=None):
  """Extracts crops from the input image tensor and resizes them.
  Extracts crops from the input image tensor and resizes them using bilinear
  sampling or nearest neighbor sampling (possibly with aspect ratio change) to a
  common output size specified by `crop_size`. This is more general than the
  `crop_to_bounding_box` op which extracts a fixed size slice from the input
  image and does not allow resizing or aspect ratio change. The crops occur
  first and then the resize.
  Returns a tensor with `crops` from the input `image` at positions defined at
  the bounding box locations in `boxes`. The cropped boxes are all resized (with
  bilinear or nearest neighbor interpolation) to a fixed
  `size = [crop_height, crop_width]`. The result is a 4-D tensor
  `[num_boxes, crop_height, crop_width, depth]`. The resizing is corner aligned.
  In particular, if `boxes = [[0, 0, 1, 1]]`, the method will give identical
  results to using `tf.compat.v1.image.resize_bilinear()` or
  `tf.compat.v1.image.resize_nearest_neighbor()`(depends on the `method`
  argument) with
  `align_corners=True`.
  Args:
    image: A 4-D tensor of shape `[batch, image_height, image_width, depth]`.
      Both `image_height` and `image_width` need to be positive.
    boxes: A 2-D tensor of shape `[num_boxes, 4]`. The `i`-th row of the tensor
      specifies the coordinates of a box in the `box_ind[i]` image and is
      specified in normalized coordinates `[y1, x1, y2, x2]`. A normalized
      coordinate value of `y` is mapped to the image coordinate at `y *
      (image_height - 1)`, so as the `[0, 1]` interval of normalized image
      height is mapped to `[0, image_height - 1]` in image height coordinates.
      We do allow `y1` > `y2`, in which case the sampled crop is an up-down
      flipped version of the original image. The width dimension is treated
      similarly. Normalized coordinates outside the `[0, 1]` range are allowed,
      in which case we use `extrapolation_value` to extrapolate the input image
      values.
    box_indices: A 1-D tensor of shape `[num_boxes]` with int32 values in `[0,
      batch)`. The value of `box_ind[i]` specifies the image that the `i`-th box
      refers to.
    crop_size: A 1-D tensor of 2 elements, `size = [crop_height, crop_width]`.
      All cropped image patches are resized to this size. The aspect ratio of
      the image content is not preserved. Both `crop_height` and `crop_width`
      need to be positive.
    method: An optional string specifying the sampling method for resizing. It
      can be either `"bilinear"` or `"nearest"` and default to `"bilinear"`.
      Currently two sampling methods are supported: Bilinear and Nearest
        Neighbor.
    extrapolation_value: An optional `float`. Defaults to `0.0`. Value used for
      extrapolation, when applicable.
    name: A name for the operation (optional).
  Returns:
    A 4-D tensor of shape `[num_boxes, crop_height, crop_width, depth]`.
  Usage example:
  >>> BATCH_SIZE = 1
  >>> NUM_BOXES = 5
  >>> IMAGE_HEIGHT = 256
  >>> IMAGE_WIDTH = 256
  >>> CHANNELS = 3
  >>> CROP_SIZE = (24, 24)
  >>> image = tf.random.normal(shape=(
  ...   BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS) )
  >>> boxes = tf.random.uniform(shape=(NUM_BOXES, 4))
  >>> box_indices = tf.random.uniform(shape=(NUM_BOXES,), minval=0,
  ...   maxval=BATCH_SIZE, dtype=tf.int32)
  >>> output = tf.image.crop_and_resize(image, boxes, box_indices, CROP_SIZE)
  >>> output.shape
  TensorShape([5, 24, 24, 3])
  Example with linear interpolation:
  >>> image = np.arange(0, 18, 2).astype('float32').reshape(3, 3)
  >>> result = tf.image.crop_and_resize(
  ...   image[None, :, :, None],
  ...   np.asarray([[0.5,0.5,1,1]]), [0], [3, 3], method='bilinear')
  >>> result[0][:, :, 0]
  <tf.Tensor: shape=(3, 3), dtype=float32, numpy=
    array([[ 8.,  9., 10.],
           [11., 12., 13.],
           [14., 15., 16.]], dtype=float32)>
  Example with nearest interpolation:
  >>> image = np.arange(0, 18, 2).astype('float32').reshape(3, 3)
  >>> result = tf.image.crop_and_resize(
  ...   image[None, :, :, None],
  ...   np.asarray([[0.5,0.5,1,1]]), [0], [3, 3], method='nearest')
  >>> result[0][:, :, 0]
  <tf.Tensor: shape=(3, 3), dtype=float32, numpy=
    array([[ 8., 10., 10.],
           [14., 16., 16.],
           [14., 16., 16.]], dtype=float32)>
  """
  return gen_image_ops.crop_and_resize(image, boxes, box_indices, crop_size,
                                       method, extrapolation_value, name)
@tf_export(v1=['image.crop_and_resize'])
@dispatch.add_dispatch_support
@deprecation.deprecated_args(None,
                             'box_ind is deprecated, use box_indices instead',
                             'box_ind')
def crop_and_resize_v1(  # pylint: disable=missing-docstring
    image,
    boxes,
    box_ind=None,
    crop_size=None,
    method='bilinear',
    extrapolation_value=0,
    name=None,
    box_indices=None):
  box_ind = deprecation.deprecated_argument_lookup('box_indices', box_indices,
                                                   'box_ind', box_ind)
  return gen_image_ops.crop_and_resize(image, boxes, box_ind, crop_size, method,
                                       extrapolation_value, name)
crop_and_resize_v1.__doc__ = gen_image_ops.crop_and_resize.__doc__
@tf_export(v1=['image.extract_glimpse'])
@dispatch.add_dispatch_support
def extract_glimpse(
    input,  # pylint: disable=redefined-builtin
    size,
    offsets,
    centered=True,
    normalized=True,
    uniform_noise=True,
    name=None):
  """Extracts a glimpse from the input tensor.
  Returns a set of windows called glimpses extracted at location
  `offsets` from the input tensor. If the windows only partially
  overlaps the inputs, the non-overlapping areas will be filled with
  random noise.
  The result is a 4-D tensor of shape `[batch_size, glimpse_height,
  glimpse_width, channels]`. The channels and batch dimensions are the
  same as that of the input tensor. The height and width of the output
  windows are specified in the `size` parameter.
  The argument `normalized` and `centered` controls how the windows are built:
  * If the coordinates are normalized but not centered, 0.0 and 1.0
    correspond to the minimum and maximum of each height and width
    dimension.
  * If the coordinates are both normalized and centered, they range from
    -1.0 to 1.0. The coordinates (-1.0, -1.0) correspond to the upper
    left corner, the lower right corner is located at (1.0, 1.0) and the
    center is at (0, 0).
  * If the coordinates are not normalized they are interpreted as
    numbers of pixels.
  Usage Example:
  >>> x = [[[[0.0],
  ...           [1.0],
  ...           [2.0]],
  ...          [[3.0],
  ...           [4.0],
  ...           [5.0]],
  ...          [[6.0],
  ...           [7.0],
  ...           [8.0]]]]
  >>> tf.compat.v1.image.extract_glimpse(x, size=(2, 2), offsets=[[1, 1]],
  ...                                    centered=False, normalized=False)
  <tf.Tensor: shape=(1, 2, 2, 1), dtype=float32, numpy=
  array([[[[0.],
           [1.]],
          [[3.],
           [4.]]]], dtype=float32)>
  Args:
    input: A `Tensor` of type `float32`. A 4-D float tensor of shape
      `[batch_size, height, width, channels]`.
    size: A `Tensor` of type `int32`. A 1-D tensor of 2 elements containing the
      size of the glimpses to extract.  The glimpse height must be specified
      first, following by the glimpse width.
    offsets: A `Tensor` of type `float32`. A 2-D integer tensor of shape
      `[batch_size, 2]` containing the y, x locations of the center of each
      window.
    centered: An optional `bool`. Defaults to `True`. indicates if the offset
      coordinates are centered relative to the image, in which case the (0, 0)
      offset is relative to the center of the input images. If false, the (0,0)
      offset corresponds to the upper left corner of the input images.
    normalized: An optional `bool`. Defaults to `True`. indicates if the offset
      coordinates are normalized.
    uniform_noise: An optional `bool`. Defaults to `True`. indicates if the
      noise should be generated using a uniform distribution or a Gaussian
      distribution.
    name: A name for the operation (optional).
  Returns:
    A `Tensor` of type `float32`.
  """
  return gen_image_ops.extract_glimpse(
      input=input,
      size=size,
      offsets=offsets,
      centered=centered,
      normalized=normalized,
      uniform_noise=uniform_noise,
      name=name)
@tf_export('image.extract_glimpse', v1=[])
@dispatch.add_dispatch_support
def extract_glimpse_v2(
    input,  # pylint: disable=redefined-builtin
    size,
    offsets,
    centered=True,
    normalized=True,
    noise='uniform',
    name=None):
  """Extracts a glimpse from the input tensor.
  Returns a set of windows called glimpses extracted at location
  `offsets` from the input tensor. If the windows only partially
  overlaps the inputs, the non-overlapping areas will be filled with
  random noise.
  The result is a 4-D tensor of shape `[batch_size, glimpse_height,
  glimpse_width, channels]`. The channels and batch dimensions are the
  same as that of the input tensor. The height and width of the output
  windows are specified in the `size` parameter.
  The argument `normalized` and `centered` controls how the windows are built:
  * If the coordinates are normalized but not centered, 0.0 and 1.0
    correspond to the minimum and maximum of each height and width
    dimension.
  * If the coordinates are both normalized and centered, they range from
    -1.0 to 1.0. The coordinates (-1.0, -1.0) correspond to the upper
    left corner, the lower right corner is located at (1.0, 1.0) and the
    center is at (0, 0).
  * If the coordinates are not normalized they are interpreted as
    numbers of pixels.
  Usage Example:
  >>> x = [[[[0.0],
  ...           [1.0],
  ...           [2.0]],
  ...          [[3.0],
  ...           [4.0],
  ...           [5.0]],
  ...          [[6.0],
  ...           [7.0],
  ...           [8.0]]]]
  >>> tf.image.extract_glimpse(x, size=(2, 2), offsets=[[1, 1]],
  ...                         centered=False, normalized=False)
  <tf.Tensor: shape=(1, 2, 2, 1), dtype=float32, numpy=
  array([[[[4.],
           [5.]],
          [[7.],
           [8.]]]], dtype=float32)>
  Args:
    input: A `Tensor` of type `float32`. A 4-D float tensor of shape
      `[batch_size, height, width, channels]`.
    size: A `Tensor` of type `int32`. A 1-D tensor of 2 elements containing the
      size of the glimpses to extract.  The glimpse height must be specified
      first, following by the glimpse width.
    offsets: A `Tensor` of type `float32`. A 2-D integer tensor of shape
      `[batch_size, 2]` containing the y, x locations of the center of each
      window.
    centered: An optional `bool`. Defaults to `True`. indicates if the offset
      coordinates are centered relative to the image, in which case the (0, 0)
      offset is relative to the center of the input images. If false, the (0,0)
      offset corresponds to the upper left corner of the input images.
    normalized: An optional `bool`. Defaults to `True`. indicates if the offset
      coordinates are normalized.
    noise: An optional `string`. Defaults to `uniform`. indicates if the noise
      should be `uniform` (uniform distribution), `gaussian` (gaussian
      distribution), or `zero` (zero padding).
    name: A name for the operation (optional).
  Returns:
    A `Tensor` of type `float32`.
  """
  return gen_image_ops.extract_glimpse_v2(
      input=input,
      size=size,
      offsets=offsets,
      centered=centered,
      normalized=normalized,
      noise=noise,
      uniform_noise=False,
      name=name)
@tf_export('image.combined_non_max_suppression')
@dispatch.add_dispatch_support
def combined_non_max_suppression(boxes,
                                 scores,
                                 max_output_size_per_class,
                                 max_total_size,
                                 iou_threshold=0.5,
                                 score_threshold=float('-inf'),
                                 pad_per_class=False,
                                 clip_boxes=True,
                                 name=None):
  """Greedily selects a subset of bounding boxes in descending order of score.
  This operation performs non_max_suppression on the inputs per batch, across
  all classes.
  Prunes away boxes that have high intersection-over-union (IOU) overlap
  with previously selected boxes.  Bounding boxes are supplied as
  [y1, x1, y2, x2], where (y1, x1) and (y2, x2) are the coordinates of any
  diagonal pair of box corners and the coordinates can be provided as normalized
  (i.e., lying in the interval [0, 1]) or absolute.  Note that this algorithm
  is agnostic to where the origin is in the coordinate system. Also note that
  this algorithm is invariant to orthogonal transformations and translations
  of the coordinate system; thus translating or reflections of the coordinate
  system result in the same boxes being selected by the algorithm.
  The output of this operation is the final boxes, scores and classes tensor
  returned after performing non_max_suppression.
  Args:
    boxes: A 4-D float `Tensor` of shape `[batch_size, num_boxes, q, 4]`. If `q`
      is 1 then same boxes are used for all classes otherwise, if `q` is equal
      to number of classes, class-specific boxes are used.
    scores: A 3-D float `Tensor` of shape `[batch_size, num_boxes, num_classes]`
      representing a single score corresponding to each box (each row of boxes).
    max_output_size_per_class: A scalar integer `Tensor` representing the
      maximum number of boxes to be selected by non-max suppression per class
    max_total_size: A int32 scalar representing maximum number of boxes retained
      over all classes. Note that setting this value to a large number may
      result in OOM error depending on the system workload.
    iou_threshold: A float representing the threshold for deciding whether boxes
      overlap too much with respect to IOU.
    score_threshold: A float representing the threshold for deciding when to
      remove boxes based on score.
    pad_per_class: If false, the output nmsed boxes, scores and classes are
      padded/clipped to `max_total_size`. If true, the output nmsed boxes,
      scores and classes are padded to be of length
      `max_size_per_class`*`num_classes`, unless it exceeds `max_total_size` in
      which case it is clipped to `max_total_size`. Defaults to false.
    clip_boxes: If true, the coordinates of output nmsed boxes will be clipped
      to [0, 1]. If false, output the box coordinates as it is. Defaults to
      true.
    name: A name for the operation (optional).
  Returns:
    'nmsed_boxes': A [batch_size, max_detections, 4] float32 tensor
      containing the non-max suppressed boxes.
    'nmsed_scores': A [batch_size, max_detections] float32 tensor containing
      the scores for the boxes.
    'nmsed_classes': A [batch_size, max_detections] float32 tensor
      containing the class for boxes.
    'valid_detections': A [batch_size] int32 tensor indicating the number of
      valid detections per batch item. Only the top valid_detections[i] entries
      in nms_boxes[i], nms_scores[i] and nms_class[i] are valid. The rest of the
      entries are zero paddings.
  """
  with ops.name_scope(name, 'combined_non_max_suppression'):
    iou_threshold = ops.convert_to_tensor(
        iou_threshold, dtype=dtypes.float32, name='iou_threshold')
    score_threshold = ops.convert_to_tensor(
        score_threshold, dtype=dtypes.float32, name='score_threshold')
    # Convert `max_total_size` to tensor *without* setting the `dtype` param.
    # This allows us to catch `int32` overflow case with `max_total_size`
    # whose expected dtype is `int32` by the op registration. Any number within
    # `int32` will get converted to `int32` tensor. Anything larger will get
    # converted to `int64`. Passing in `int64` for `max_total_size` to the op
    # will throw dtype mismatch exception.
    # TODO(b/173251596): Once there is a more general solution to warn against
    # int overflow conversions, revisit this check.
    max_total_size = ops.convert_to_tensor(max_total_size)
    return gen_image_ops.combined_non_max_suppression(
        boxes, scores, max_output_size_per_class, max_total_size, iou_threshold,
        score_threshold, pad_per_class, clip_boxes)
def _bbox_overlap(boxes_a, boxes_b):
  """Calculates the overlap (iou - intersection over union) between boxes_a and boxes_b.
  Args:
    boxes_a: a tensor with a shape of [batch_size, N, 4]. N is the number of
      boxes per image. The last dimension is the pixel coordinates in
      [ymin, xmin, ymax, xmax] form.
    boxes_b: a tensor with a shape of [batch_size, M, 4]. M is the number of
      boxes. The last dimension is the pixel coordinates in
      [ymin, xmin, ymax, xmax] form.
  Returns:
    intersection_over_union: a tensor with as a shape of [batch_size, N, M],
    representing the ratio of intersection area over union area (IoU) between
    two boxes
  """
  with ops.name_scope('bbox_overlap'):
    a_y_min, a_x_min, a_y_max, a_x_max = array_ops.split(
        value=boxes_a, num_or_size_splits=4, axis=2)
    b_y_min, b_x_min, b_y_max, b_x_max = array_ops.split(
        value=boxes_b, num_or_size_splits=4, axis=2)
    # Calculates the intersection area.
    i_xmin = math_ops.maximum(
        a_x_min, array_ops.transpose(b_x_min, [0, 2, 1]))
    i_xmax = math_ops.minimum(
        a_x_max, array_ops.transpose(b_x_max, [0, 2, 1]))
    i_ymin = math_ops.maximum(
        a_y_min, array_ops.transpose(b_y_min, [0, 2, 1]))
    i_ymax = math_ops.minimum(
        a_y_max, array_ops.transpose(b_y_max, [0, 2, 1]))
    i_area = math_ops.maximum(
        (i_xmax - i_xmin), 0) * math_ops.maximum((i_ymax - i_ymin), 0)
    # Calculates the union area.
    a_area = (a_y_max - a_y_min) * (a_x_max - a_x_min)
    b_area = (b_y_max - b_y_min) * (b_x_max - b_x_min)
    EPSILON = 1e-8
    # Adds a small epsilon to avoid divide-by-zero.
    u_area = a_area + array_ops.transpose(b_area, [0, 2, 1]) - i_area + EPSILON
    # Calculates IoU.
    intersection_over_union = i_area / u_area
    return intersection_over_union
def _self_suppression(iou, _, iou_sum, iou_threshold):
  """Suppress boxes in the same tile.
     Compute boxes that cannot be suppressed by others (i.e.,
     can_suppress_others), and then use them to suppress boxes in the same tile.
  Args:
    iou: a tensor of shape [batch_size, num_boxes_with_padding] representing
    intersection over union.
    iou_sum: a scalar tensor.
    iou_threshold: a scalar tensor.
  Returns:
    iou_suppressed: a tensor of shape [batch_size, num_boxes_with_padding].
    iou_diff: a scalar tensor representing whether any box is supressed in
      this step.
    iou_sum_new: a scalar tensor of shape [batch_size] that represents
      the iou sum after suppression.
    iou_threshold: a scalar tensor.
  """
  batch_size = array_ops.shape(iou)[0]
  can_suppress_others = math_ops.cast(
      array_ops.reshape(
          math_ops.reduce_max(iou, 1) < iou_threshold, [batch_size, -1, 1]),
      iou.dtype)
  iou_after_suppression = array_ops.reshape(
      math_ops.cast(
          math_ops.reduce_max(can_suppress_others * iou, 1) < iou_threshold,
          iou.dtype),
      [batch_size, -1, 1]) * iou
  iou_sum_new = math_ops.reduce_sum(iou_after_suppression, [1, 2])
  return [
      iou_after_suppression,
      math_ops.reduce_any(iou_sum - iou_sum_new > iou_threshold), iou_sum_new,
      iou_threshold
  ]
def _cross_suppression(boxes, box_slice, iou_threshold, inner_idx, tile_size):
  """Suppress boxes between different tiles.
  Args:
    boxes: a tensor of shape [batch_size, num_boxes_with_padding, 4]
    box_slice: a tensor of shape [batch_size, tile_size, 4]
    iou_threshold: a scalar tensor
    inner_idx: a scalar tensor representing the tile index of the tile
      that is used to supress box_slice
    tile_size: an integer representing the number of boxes in a tile
  Returns:
    boxes: unchanged boxes as input
    box_slice_after_suppression: box_slice after suppression
    iou_threshold: unchanged
  """
  batch_size = array_ops.shape(boxes)[0]
  new_slice = array_ops.slice(
      boxes, [0, inner_idx * tile_size, 0],
      [batch_size, tile_size, 4])
  iou = _bbox_overlap(new_slice, box_slice)
  box_slice_after_suppression = array_ops.expand_dims(
      math_ops.cast(math_ops.reduce_all(iou < iou_threshold, [1]),
                    box_slice.dtype),
      2) * box_slice
  return boxes, box_slice_after_suppression, iou_threshold, inner_idx + 1
def _suppression_loop_body(boxes, iou_threshold, output_size, idx, tile_size):
  """Process boxes in the range [idx*tile_size, (idx+1)*tile_size).
  Args:
    boxes: a tensor with a shape of [batch_size, anchors, 4].
    iou_threshold: a float representing the threshold for deciding whether boxes
      overlap too much with respect to IOU.
    output_size: an int32 tensor of size [batch_size]. Representing the number
      of selected boxes for each batch.
    idx: an integer scalar representing induction variable.
    tile_size: an integer representing the number of boxes in a tile
  Returns:
    boxes: updated boxes.
    iou_threshold: pass down iou_threshold to the next iteration.
    output_size: the updated output_size.
    idx: the updated induction variable.
  """
  with ops.name_scope('suppression_loop_body'):
    num_tiles = array_ops.shape(boxes)[1] // tile_size
    batch_size = array_ops.shape(boxes)[0]
    def cross_suppression_func(boxes, box_slice, iou_threshold, inner_idx):
      return _cross_suppression(boxes, box_slice, iou_threshold, inner_idx,
                                tile_size)
    # Iterates over tiles that can possibly suppress the current tile.
    box_slice = array_ops.slice(boxes, [0, idx * tile_size, 0],
                                [batch_size, tile_size, 4])
    _, box_slice, _, _ = while_loop.while_loop(
        lambda _boxes, _box_slice, _threshold, inner_idx: inner_idx < idx,
        cross_suppression_func,
        [boxes, box_slice, iou_threshold,
         constant_op.constant(0)])
    # Iterates over the current tile to compute self-suppression.
    iou = _bbox_overlap(box_slice, box_slice)
    mask = array_ops.expand_dims(
        array_ops.reshape(
            math_ops.range(tile_size), [1, -1]) > array_ops.reshape(
                math_ops.range(tile_size), [-1, 1]), 0)
    iou *= math_ops.cast(
        math_ops.logical_and(mask, iou >= iou_threshold), iou.dtype)
    suppressed_iou, _, _, _ = while_loop.while_loop(
        lambda _iou, loop_condition, _iou_sum, _: loop_condition,
        _self_suppression, [
            iou,
            constant_op.constant(True),
            math_ops.reduce_sum(iou, [1, 2]), iou_threshold
        ])
    suppressed_box = math_ops.reduce_sum(suppressed_iou, 1) > 0
    box_slice *= array_ops.expand_dims(
        1.0 - math_ops.cast(suppressed_box, box_slice.dtype), 2)
    # Uses box_slice to update the input boxes.
    mask = array_ops.reshape(
        math_ops.cast(
            math_ops.equal(math_ops.range(num_tiles), idx), boxes.dtype),
        [1, -1, 1, 1])
    boxes = array_ops.tile(array_ops.expand_dims(
        box_slice, [1]), [1, num_tiles, 1, 1]) * mask + array_ops.reshape(
            boxes, [batch_size, num_tiles, tile_size, 4]) * (1 - mask)
    boxes = array_ops.reshape(boxes, [batch_size, -1, 4])
    # Updates output_size.
    output_size += math_ops.reduce_sum(
        math_ops.cast(
            math_ops.reduce_any(box_slice > 0, [2]), dtypes.int32), [1])
  return boxes, iou_threshold, output_size, idx + 1
@tf_export('image.non_max_suppression_padded')
@dispatch.add_dispatch_support
def non_max_suppression_padded(boxes,
                               scores,
                               max_output_size,
                               iou_threshold=0.5,
                               score_threshold=float('-inf'),
                               pad_to_max_output_size=False,
                               name=None,
                               sorted_input=False,
                               canonicalized_coordinates=False,
                               tile_size=512):
  """Greedily selects a subset of bounding boxes in descending order of score.
  Performs algorithmically equivalent operation to tf.image.non_max_suppression,
  with the addition of an optional parameter which zero-pads the output to
  be of size `max_output_size`.
  The output of this operation is a tuple containing the set of integers
  indexing into the input collection of bounding boxes representing the selected
  boxes and the number of valid indices in the index set.  The bounding box
  coordinates corresponding to the selected indices can then be obtained using
  the `tf.slice` and `tf.gather` operations.  For example:
    ```python
    selected_indices_padded, num_valid = tf.image.non_max_suppression_padded(
        boxes, scores, max_output_size, iou_threshold,
        score_threshold, pad_to_max_output_size=True)
    selected_indices = tf.slice(
        selected_indices_padded, tf.constant([0]), num_valid)
    selected_boxes = tf.gather(boxes, selected_indices)
    ```
  Args:
    boxes: a tensor of rank 2 or higher with a shape of [..., num_boxes, 4].
      Dimensions except the last two are batch dimensions.
    scores: a tensor of rank 1 or higher with a shape of [..., num_boxes].
    max_output_size: a scalar integer `Tensor` representing the maximum number
      of boxes to be selected by non max suppression. Note that setting this
      value to a large number may result in OOM error depending on the system
      workload.
    iou_threshold: a float representing the threshold for deciding whether boxes
      overlap too much with respect to IoU (intersection over union).
    score_threshold: a float representing the threshold for box scores. Boxes
      with a score that is not larger than this threshold will be suppressed.
    pad_to_max_output_size: whether to pad the output idx to max_output_size.
      Must be set to True when the input is a batch of images.
    name: name of operation.
    sorted_input: a boolean indicating whether the input boxes and scores
      are sorted in descending order by the score.
    canonicalized_coordinates: if box coordinates are given as
    `[y_min, x_min, y_max, x_max]`, setting to True eliminate redundant
     computation to canonicalize box coordinates.
    tile_size: an integer representing the number of boxes in a tile, i.e.,
      the maximum number of boxes per image that can be used to suppress other
      boxes in parallel; larger tile_size means larger parallelism and
      potentially more redundant work.
  Returns:
    idx: a tensor with a shape of [..., num_boxes] representing the
      indices selected by non-max suppression. The leading dimensions
      are the batch dimensions of the input boxes. All numbers are within
      [0, num_boxes). For each image (i.e., idx[i]), only the first num_valid[i]
      indices (i.e., idx[i][:num_valid[i]]) are valid.
    num_valid: a tensor of rank 0 or higher with a shape of [...]
      representing the number of valid indices in idx. Its dimensions are the
      batch dimensions of the input boxes.
   Raises:
    ValueError: When set pad_to_max_output_size to False for batched input.
  """
  with ops.name_scope(name, 'non_max_suppression_padded'):
    if not pad_to_max_output_size:
      # pad_to_max_output_size may be set to False only when the shape of
      # boxes is [num_boxes, 4], i.e., a single image. We make best effort to
      # detect violations at compile time. If `boxes` does not have a static
      # rank, the check allows computation to proceed.
      if boxes.get_shape().rank is not None and boxes.get_shape().rank > 2:
        raise ValueError("'pad_to_max_output_size' (value {}) must be True for "
                         'batched input'.format(pad_to_max_output_size))
    if name is None:
      name = ''
    idx, num_valid = non_max_suppression_padded_v2(
        boxes, scores, max_output_size, iou_threshold, score_threshold,
        sorted_input, canonicalized_coordinates, tile_size)
    # def_function.function seems to lose shape information, so set it here.
    if not pad_to_max_output_size:
      idx = idx[0, :num_valid]
    else:
      batch_dims = array_ops.concat([
          array_ops.shape(boxes)[:-2],
          array_ops.expand_dims(max_output_size, 0)
      ], 0)
      idx = array_ops.reshape(idx, batch_dims)
    return idx, num_valid
# TODO(b/158709815): Improve performance regression due to
# def_function.function.
@def_function.function(
    experimental_implements='non_max_suppression_padded_v2')
def non_max_suppression_padded_v2(boxes,
                                  scores,
                                  max_output_size,
                                  iou_threshold=0.5,
                                  score_threshold=float('-inf'),
                                  sorted_input=False,
                                  canonicalized_coordinates=False,
                                  tile_size=512):
  """Non-maximum suppression.
  Prunes away boxes that have high intersection-over-union (IOU) overlap
  with previously selected boxes. Bounding boxes are supplied as
  `[y1, x1, y2, x2]`, where `(y1, x1)` and `(y2, x2)` are the coordinates of any
  diagonal pair of box corners and the coordinates can be provided as normalized
  (i.e., lying in the interval `[0, 1]`) or absolute. The bounding box
  coordinates are cannonicalized to `[y_min, x_min, y_max, x_max]`,
  where `(y_min, x_min)` and `(y_max, x_mas)` are the coordinates of the lower
  left and upper right corner. User may indiciate the input box coordinates are
  already canonicalized to eliminate redundant work by setting
  canonicalized_coordinates to `True`. Note that this algorithm is agnostic to
  where the origin is in the coordinate system. Note that this algorithm is
  invariant to orthogonal transformations and translations of the coordinate
  system; thus translating or reflections of the coordinate system result in the
  same boxes being selected by the algorithm.
  Similar to tf.image.non_max_suppression, non_max_suppression_padded
  implements hard NMS but can operate on a batch of images and improves
  performance by titling the bounding boxes. Non_max_suppression_padded should
  be preferred over tf.image_non_max_suppression when running on devices with
  abundant parallelsim for higher computation speed. For soft NMS, refer to
  tf.image.non_max_suppression_with_scores.
  While a serial NMS algorithm iteratively uses the highest-scored unprocessed
  box to suppress boxes, this algorithm uses many boxes to suppress other boxes
  in parallel. The key idea is to partition boxes into tiles based on their
  score and suppresses boxes tile by tile, thus achieving parallelism within a
  tile. The tile size determines the degree of parallelism.
  In cross suppression (using boxes of tile A to suppress boxes of tile B),
  all boxes in A can independently suppress boxes in B.
  Self suppression (suppressing boxes of the same tile) needs to be iteratively
  applied until there's no more suppression. In each iteration, boxes that
  cannot be suppressed are used to suppress boxes in the same tile.
  boxes = boxes.pad_to_multiply_of(tile_size)
  num_tiles = len(boxes) // tile_size
  output_boxes = []
  for i in range(num_tiles):
    box_tile = boxes[i*tile_size : (i+1)*tile_size]
    for j in range(i - 1):
      # in parallel suppress boxes in box_tile using boxes from suppressing_tile
      suppressing_tile = boxes[j*tile_size : (j+1)*tile_size]
      iou = _bbox_overlap(box_tile, suppressing_tile)
      # if the box is suppressed in iou, clear it to a dot
      box_tile *= _update_boxes(iou)
    # Iteratively handle the diagnal tile.
    iou = _box_overlap(box_tile, box_tile)
    iou_changed = True
    while iou_changed:
      # boxes that are not suppressed by anything else
      suppressing_boxes = _get_suppressing_boxes(iou)
      # boxes that are suppressed by suppressing_boxes
      suppressed_boxes = _get_suppressed_boxes(iou, suppressing_boxes)
      # clear iou to 0 for boxes that are suppressed, as they cannot be used
      # to suppress other boxes any more
      new_iou = _clear_iou(iou, suppressed_boxes)
      iou_changed = (new_iou != iou)
      iou = new_iou
    # remaining boxes that can still suppress others, are selected boxes.
    output_boxes.append(_get_suppressing_boxes(iou))
    if len(output_boxes) >= max_output_size:
      break
  Args:
    boxes: a tensor of rank 2 or higher with a shape of [..., num_boxes, 4].
      Dimensions except the last two are batch dimensions. The last dimension
      represents box coordinates, given as [y_1, x_1, y_2, x_2]. The coordinates
      on each dimension can be given in any order
      (see also `canonicalized_coordinates`) but must describe a box with
      a positive area.
    scores: a tensor of rank 1 or higher with a shape of [..., num_boxes].
    max_output_size: a scalar integer `Tensor` representing the maximum number
      of boxes to be selected by non max suppression.
    iou_threshold: a float representing the threshold for deciding whether boxes
      overlap too much with respect to IoU (intersection over union).
    score_threshold: a float representing the threshold for box scores. Boxes
      with a score that is not larger than this threshold will be suppressed.
    sorted_input: a boolean indicating whether the input boxes and scores
      are sorted in descending order by the score.
    canonicalized_coordinates: if box coordinates are given as
    `[y_min, x_min, y_max, x_max]`, setting to True eliminate redundant
     computation to canonicalize box coordinates.
    tile_size: an integer representing the number of boxes in a tile, i.e.,
      the maximum number of boxes per image that can be used to suppress other
      boxes in parallel; larger tile_size means larger parallelism and
      potentially more redundant work.
  Returns:
    idx: a tensor with a shape of [..., num_boxes] representing the
      indices selected by non-max suppression. The leading dimensions
      are the batch dimensions of the input boxes. All numbers are within
      [0, num_boxes). For each image (i.e., idx[i]), only the first num_valid[i]
      indices (i.e., idx[i][:num_valid[i]]) are valid.
    num_valid: a tensor of rank 0 or higher with a shape of [...]
      representing the number of valid indices in idx. Its dimensions are the
      batch dimensions of the input boxes.
   Raises:
    ValueError: When set pad_to_max_output_size to False for batched input.
  """
  def _sort_scores_and_boxes(scores, boxes):
    """Sort boxes based their score from highest to lowest.
    Args:
      scores: a tensor with a shape of [batch_size, num_boxes] representing
        the scores of boxes.
      boxes: a tensor with a shape of [batch_size, num_boxes, 4] representing
        the boxes.
    Returns:
      sorted_scores: a tensor with a shape of [batch_size, num_boxes]
        representing the sorted scores.
      sorted_boxes: a tensor representing the sorted boxes.
      sorted_scores_indices: a tensor with a shape of [batch_size, num_boxes]
        representing the index of the scores in a sorted descending order.
    """
    with ops.name_scope('sort_scores_and_boxes'):
      sorted_scores_indices = sort_ops.argsort(
          scores, axis=1, direction='DESCENDING')
      sorted_scores = array_ops.gather(
          scores, sorted_scores_indices, axis=1, batch_dims=1
      )
      sorted_boxes = array_ops.gather(
          boxes, sorted_scores_indices, axis=1, batch_dims=1
      )
    return sorted_scores, sorted_boxes, sorted_scores_indices
  batch_dims = array_ops.shape(boxes)[:-2]
  num_boxes = array_ops.shape(boxes)[-2]
  boxes = array_ops.reshape(boxes, [-1, num_boxes, 4])
  scores = array_ops.reshape(scores, [-1, num_boxes])
  batch_size = array_ops.shape(boxes)[0]
  if score_threshold != float('-inf'):
    with ops.name_scope('filter_by_score'):
      score_mask = math_ops.cast(scores > score_threshold, scores.dtype)
      scores *= score_mask
      box_mask = array_ops.expand_dims(
          math_ops.cast(score_mask, boxes.dtype), 2)
      boxes *= box_mask
  if not canonicalized_coordinates:
    with ops.name_scope('canonicalize_coordinates'):
      y_1, x_1, y_2, x_2 = array_ops.split(
          value=boxes, num_or_size_splits=4, axis=2)
      y_1_is_min = math_ops.reduce_all(
          math_ops.less_equal(y_1[0, 0, 0], y_2[0, 0, 0]))
      y_min, y_max = tf_cond.cond(
          y_1_is_min, lambda: (y_1, y_2), lambda: (y_2, y_1))
      x_1_is_min = math_ops.reduce_all(
          math_ops.less_equal(x_1[0, 0, 0], x_2[0, 0, 0]))
      x_min, x_max = tf_cond.cond(
          x_1_is_min, lambda: (x_1, x_2), lambda: (x_2, x_1))
      boxes = array_ops.concat([y_min, x_min, y_max, x_max], axis=2)
  # TODO(@bhack): https://github.com/tensorflow/tensorflow/issues/56089
  # this will be required after deprecation
  #else:
  #  y_1, x_1, y_2, x_2 = array_ops.split(
  #      value=boxes, num_or_size_splits=4, axis=2)
  if not sorted_input:
    scores, boxes, sorted_indices = _sort_scores_and_boxes(scores, boxes)
  else:
    # Default value required for Autograph.
    sorted_indices = array_ops.zeros_like(scores, dtype=dtypes.int32)
  pad = math_ops.cast(
      math_ops.ceil(
          math_ops.cast(
              math_ops.maximum(num_boxes, max_output_size), dtypes.float32) /
          math_ops.cast(tile_size, dtypes.float32)),
      dtypes.int32) * tile_size - num_boxes
  boxes = array_ops.pad(
      math_ops.cast(boxes, dtypes.float32), [[0, 0], [0, pad], [0, 0]])
  scores = array_ops.pad(
      math_ops.cast(scores, dtypes.float32), [[0, 0], [0, pad]])
  num_boxes_after_padding = num_boxes + pad
  num_iterations = num_boxes_after_padding // tile_size
  def _loop_cond(unused_boxes, unused_threshold, output_size, idx):
    return math_ops.logical_and(
        math_ops.reduce_min(output_size) < max_output_size,
        idx < num_iterations)
  def suppression_loop_body(boxes, iou_threshold, output_size, idx):
    return _suppression_loop_body(
        boxes, iou_threshold, output_size, idx, tile_size)
  selected_boxes, _, output_size, _ = while_loop.while_loop(
      _loop_cond,
      suppression_loop_body,
      [
          boxes, iou_threshold,
          array_ops.zeros([batch_size], dtypes.int32),
          constant_op.constant(0)
      ],
      shape_invariants=[
          tensor_shape.TensorShape([None, None, 4]),
          tensor_shape.TensorShape([]),
          tensor_shape.TensorShape([None]),
          tensor_shape.TensorShape([]),
      ],
  )
  num_valid = math_ops.minimum(output_size, max_output_size)
  idx = num_boxes_after_padding - math_ops.cast(
      nn_ops.top_k(
          math_ops.cast(math_ops.reduce_any(
              selected_boxes > 0, [2]), dtypes.int32) *
          array_ops.expand_dims(
              math_ops.range(num_boxes_after_padding, 0, -1), 0),
          max_output_size)[0], dtypes.int32)
  idx = math_ops.minimum(idx, num_boxes - 1)
  if not sorted_input:
    index_offsets = math_ops.range(batch_size) * num_boxes
    gather_idx = array_ops.reshape(
        idx + array_ops.expand_dims(index_offsets, 1), [-1])
    idx = array_ops.reshape(
        array_ops.gather(array_ops.reshape(sorted_indices, [-1]),
                         gather_idx),
        [batch_size, -1])
  invalid_index = array_ops.zeros([batch_size, max_output_size],
                                  dtype=dtypes.int32)
  idx_index = array_ops.expand_dims(math_ops.range(max_output_size), 0)
  num_valid_expanded = array_ops.expand_dims(num_valid, 1)
  idx = array_ops.where(idx_index < num_valid_expanded,
                        idx, invalid_index)
  num_valid = array_ops.reshape(num_valid, batch_dims)
  return idx, num_valid
def non_max_suppression_padded_v1(boxes,
                                  scores,
                                  max_output_size,
                                  iou_threshold=0.5,
                                  score_threshold=float('-inf'),
                                  pad_to_max_output_size=False,
                                  name=None):
  """Greedily selects a subset of bounding boxes in descending order of score.
  Performs algorithmically equivalent operation to tf.image.non_max_suppression,
  with the addition of an optional parameter which zero-pads the output to
  be of size `max_output_size`.
  The output of this operation is a tuple containing the set of integers
  indexing into the input collection of bounding boxes representing the selected
  boxes and the number of valid indices in the index set.  The bounding box
  coordinates corresponding to the selected indices can then be obtained using
  the `tf.slice` and `tf.gather` operations.  For example:
    ```python
    selected_indices_padded, num_valid = tf.image.non_max_suppression_padded(
        boxes, scores, max_output_size, iou_threshold,
        score_threshold, pad_to_max_output_size=True)
    selected_indices = tf.slice(
        selected_indices_padded, tf.constant([0]), num_valid)
    selected_boxes = tf.gather(boxes, selected_indices)
    ```
  Args:
    boxes: A 2-D float `Tensor` of shape `[num_boxes, 4]`.
    scores: A 1-D float `Tensor` of shape `[num_boxes]` representing a single
      score corresponding to each box (each row of boxes).
    max_output_size: A scalar integer `Tensor` representing the maximum number
      of boxes to be selected by non-max suppression.
    iou_threshold: A float representing the threshold for deciding whether boxes
      overlap too much with respect to IOU.
    score_threshold: A float representing the threshold for deciding when to
      remove boxes based on score.
    pad_to_max_output_size: bool.  If True, size of `selected_indices` output is
      padded to `max_output_size`.
    name: A name for the operation (optional).
  Returns:
    selected_indices: A 1-D integer `Tensor` of shape `[M]` representing the
      selected indices from the boxes tensor, where `M <= max_output_size`.
    valid_outputs: A scalar integer `Tensor` denoting how many elements in
    `selected_indices` are valid.  Valid elements occur first, then padding.
  """
  with ops.name_scope(name, 'non_max_suppression_padded'):
    iou_threshold = ops.convert_to_tensor(iou_threshold, name='iou_threshold')
    score_threshold = ops.convert_to_tensor(
        score_threshold, name='score_threshold')
    return gen_image_ops.non_max_suppression_v4(boxes, scores, max_output_size,
                                                iou_threshold, score_threshold,
                                                pad_to_max_output_size)
@tf_export('image.draw_bounding_boxes', v1=[])
@dispatch.add_dispatch_support
def draw_bounding_boxes_v2(images, boxes, colors, name=None):
  """Draw bounding boxes on a batch of images.
  Outputs a copy of `images` but draws on top of the pixels zero or more
  bounding boxes specified by the locations in `boxes`. The coordinates of the
  each bounding box in `boxes` are encoded as `[y_min, x_min, y_max, x_max]`.
  The bounding box coordinates are floats in `[0.0, 1.0]` relative to the width
  and the height of the underlying image.
  For example, if an image is 100 x 200 pixels (height x width) and the bounding
  box is `[0.1, 0.2, 0.5, 0.9]`, the upper-left and bottom-right coordinates of
  the bounding box will be `(40, 10)` to `(180, 50)` (in (x,y) coordinates).
  Parts of the bounding box may fall outside the image.
  Args:
    images: A `Tensor`. Must be one of the following types: `float32`, `half`.
      4-D with shape `[batch, height, width, depth]`. A batch of images.
    boxes: A `Tensor` of type `float32`. 3-D with shape `[batch,
      num_bounding_boxes, 4]` containing bounding boxes.
    colors: A `Tensor` of type `float32`. 2-D. A list of RGBA colors to cycle
      through for the boxes.
    name: A name for the operation (optional).
  Returns:
    A `Tensor`. Has the same type as `images`.
  Usage Example:
  >>> # create an empty image
  >>> img = tf.zeros([1, 3, 3, 3])
  >>> # draw a box around the image
  >>> box = np.array([0, 0, 1, 1])
  >>> boxes = box.reshape([1, 1, 4])
  >>> # alternate between red and blue
  >>> colors = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
  >>> tf.image.draw_bounding_boxes(img, boxes, colors)
  <tf.Tensor: shape=(1, 3, 3, 3), dtype=float32, numpy=
  array([[[[1., 0., 0.],
          [1., 0., 0.],
          [1., 0., 0.]],
          [[1., 0., 0.],
          [0., 0., 0.],
          [1., 0., 0.]],
          [[1., 0., 0.],
          [1., 0., 0.],
          [1., 0., 0.]]]], dtype=float32)>
  """
  if colors is None:
    return gen_image_ops.draw_bounding_boxes(images, boxes, name)
  return gen_image_ops.draw_bounding_boxes_v2(images, boxes, colors, name)
@tf_export(v1=['image.draw_bounding_boxes'])
@dispatch.add_dispatch_support
def draw_bounding_boxes(images, boxes, name=None, colors=None):
  """Draw bounding boxes on a batch of images.
  Outputs a copy of `images` but draws on top of the pixels zero or more
  bounding boxes specified by the locations in `boxes`. The coordinates of the
  each bounding box in `boxes` are encoded as `[y_min, x_min, y_max, x_max]`.
  The bounding box coordinates are floats in `[0.0, 1.0]` relative to the width
  and the height of the underlying image.
  For example, if an image is 100 x 200 pixels (height x width) and the bounding
  box is `[0.1, 0.2, 0.5, 0.9]`, the upper-left and bottom-right coordinates of
  the bounding box will be `(40, 10)` to `(180, 50)` (in (x,y) coordinates).
  Parts of the bounding box may fall outside the image.
  Args:
    images: A `Tensor`. Must be one of the following types: `float32`, `half`.
      4-D with shape `[batch, height, width, depth]`. A batch of images.
    boxes: A `Tensor` of type `float32`. 3-D with shape `[batch,
      num_bounding_boxes, 4]` containing bounding boxes.
    name: A name for the operation (optional).
    colors: A `Tensor` of type `float32`. 2-D. A list of RGBA colors to cycle
      through for the boxes.
  Returns:
    A `Tensor`. Has the same type as `images`.
  Usage Example:
  >>> # create an empty image
  >>> img = tf.zeros([1, 3, 3, 3])
  >>> # draw a box around the image
  >>> box = np.array([0, 0, 1, 1])
  >>> boxes = box.reshape([1, 1, 4])
  >>> # alternate between red and blue
  >>> colors = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
  >>> tf.image.draw_bounding_boxes(img, boxes, colors)
  <tf.Tensor: shape=(1, 3, 3, 3), dtype=float32, numpy=
  array([[[[1., 0., 0.],
          [1., 0., 0.],
          [1., 0., 0.]],
          [[1., 0., 0.],
          [0., 0., 0.],
          [1., 0., 0.]],
          [[1., 0., 0.],
          [1., 0., 0.],
          [1., 0., 0.]]]], dtype=float32)>
  """
  return draw_bounding_boxes_v2(images, boxes, colors, name)
@tf_export('image.generate_bounding_box_proposals')
@dispatch.add_dispatch_support
def generate_bounding_box_proposals(scores,
                                    bbox_deltas,
                                    image_info,
                                    anchors,
                                    nms_threshold=0.7,
                                    pre_nms_topn=6000,
                                    min_size=16,
                                    post_nms_topn=300,
                                    name=None):
  """Generate bounding box proposals from encoded bounding boxes.
  Args:
    scores: A 4-D float `Tensor` of shape
     `[num_images, height, width, num_achors]` containing scores of
      the boxes for given anchors, can be unsorted.
    bbox_deltas: A 4-D float `Tensor` of shape
     `[num_images, height, width, 4 x num_anchors]` encoding boxes
      with respect to each anchor. Coordinates are given
      in the form `[dy, dx, dh, dw]`.
    image_info: A 2-D float `Tensor` of shape `[num_images, 5]`
      containing image information Height, Width, Scale.
    anchors: A 2-D float `Tensor` of shape `[num_anchors, 4]`
      describing the anchor boxes.
      Boxes are formatted in the form `[y1, x1, y2, x2]`.
    nms_threshold: A scalar float `Tensor` for non-maximal-suppression
      threshold. Defaults to 0.7.
    pre_nms_topn: A scalar int `Tensor` for the number of
      top scoring boxes to be used as input. Defaults to 6000.
    min_size: A scalar float `Tensor`. Any box that has a smaller size
      than min_size will be discarded. Defaults to 16.
    post_nms_topn: An integer. Maximum number of rois in the output.
    name: A name for this operation (optional).
  Returns:
    rois: Region of interest boxes sorted by their scores.
    roi_probabilities: scores of the ROI boxes in the ROIs' `Tensor`.
  """
  return gen_image_ops.generate_bounding_box_proposals(
      scores=scores,
      bbox_deltas=bbox_deltas,
      image_info=image_info,
      anchors=anchors,
      nms_threshold=nms_threshold,
      pre_nms_topn=pre_nms_topn,
      min_size=min_size,
      post_nms_topn=post_nms_topn,
      name=name)
