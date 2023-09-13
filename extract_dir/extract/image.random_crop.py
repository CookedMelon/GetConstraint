@tf_export("image.random_crop", v1=["image.random_crop", "random_crop"])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints("random_crop")
def random_crop(value, size, seed=None, name=None):
  """Randomly crops a tensor to a given size.
  Slices a shape `size` portion out of `value` at a uniformly chosen offset.
  Requires `value.shape >= size`.
  If a dimension should not be cropped, pass the full size of that dimension.
  For example, RGB images can be cropped with
  `size = [crop_height, crop_width, 3]`.
  Example usage:
  >>> image = [[1, 2, 3], [4, 5, 6]]
  >>> result = tf.image.random_crop(value=image, size=(1, 3))
  >>> result.shape.as_list()
  [1, 3]
  For producing deterministic results given a `seed` value, use
  `tf.image.stateless_random_crop`. Unlike using the `seed` param with
  `tf.image.random_*` ops, `tf.image.stateless_random_*` ops guarantee the same
  results given the same seed independent of how many times the function is
  called, and independent of global seed settings (e.g. tf.random.set_seed).
  Args:
    value: Input tensor to crop.
    size: 1-D tensor with size the rank of `value`.
    seed: Python integer. Used to create a random seed. See
      `tf.random.set_seed`
      for behavior.
    name: A name for this operation (optional).
  Returns:
    A cropped tensor of the same rank as `value` and shape `size`.
  """
  with ops.name_scope(name, "random_crop", [value, size]) as name:
    value = ops.convert_to_tensor(value, name="value")
    size = ops.convert_to_tensor(size, dtype=dtypes.int32, name="size")
    shape = array_ops.shape(value)
    check = control_flow_assert.Assert(
        math_ops.reduce_all(shape >= size),
        ["Need value.shape >= size, got ", shape, size],
        summarize=1000)
    shape = control_flow_ops.with_dependencies([check], shape)
    limit = shape - size + 1
    offset = random_uniform(
        array_ops.shape(shape),
        dtype=size.dtype,
        maxval=size.dtype.max,
        seed=seed) % limit
    return array_ops.slice(value, offset, size, name=name)
