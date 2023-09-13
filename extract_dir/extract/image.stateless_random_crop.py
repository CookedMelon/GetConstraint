@tf_export("image.stateless_random_crop", v1=[])
@dispatch.add_dispatch_support
def stateless_random_crop(value, size, seed, name=None):
  """Randomly crops a tensor to a given size in a deterministic manner.
  Slices a shape `size` portion out of `value` at a uniformly chosen offset.
  Requires `value.shape >= size`.
  If a dimension should not be cropped, pass the full size of that dimension.
  For example, RGB images can be cropped with
  `size = [crop_height, crop_width, 3]`.
  Guarantees the same results given the same `seed` independent of how many
  times the function is called, and independent of global seed settings (e.g.
  `tf.random.set_seed`).
  Usage Example:
  >>> image = [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]
  >>> seed = (1, 2)
  >>> tf.image.stateless_random_crop(value=image, size=(1, 2, 3), seed=seed)
  <tf.Tensor: shape=(1, 2, 3), dtype=int32, numpy=
  array([[[1, 2, 3],
          [4, 5, 6]]], dtype=int32)>
  Args:
    value: Input tensor to crop.
    size: 1-D tensor with size the rank of `value`.
    seed: A shape [2] Tensor, the seed to the random number generator. Must have
      dtype `int32` or `int64`. (When using XLA, only `int32` is allowed.)
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
    offset = stateless_random_ops.stateless_random_uniform(
        array_ops.shape(shape),
        dtype=size.dtype,
        maxval=size.dtype.max,
        seed=seed) % limit
    return array_ops.slice(value, offset, size, name=name)
