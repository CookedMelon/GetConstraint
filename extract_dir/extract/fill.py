@tf_export("fill")
@dispatch.add_dispatch_support
def fill(dims, value, name=None):
  r"""Creates a tensor filled with a scalar value.
  See also `tf.ones`, `tf.zeros`, `tf.one_hot`, `tf.eye`.
  This operation creates a tensor of shape `dims` and fills it with `value`.
  For example:
  >>> tf.fill([2, 3], 9)
  <tf.Tensor: shape=(2, 3), dtype=int32, numpy=
  array([[9, 9, 9],
         [9, 9, 9]], dtype=int32)>
  `tf.fill` evaluates at graph runtime and supports dynamic shapes based on
  other runtime `tf.Tensors`, unlike `tf.constant(value, shape=dims)`, which
  embeds the value as a `Const` node.
  Args:
    dims: A 1-D sequence of non-negative numbers. Represents the shape of the
      output `tf.Tensor`. Entries should be of type: `int32`, `int64`.
    value: A value to fill the returned `tf.Tensor`.
    name: Optional string. The name of the output `tf.Tensor`.
  Returns:
    A `tf.Tensor` with shape `dims` and the same dtype as `value`.
  Raises:
    InvalidArgumentError: `dims` contains negative entries.
    NotFoundError: `dims` contains non-integer entries.
  @compatibility(numpy)
  Similar to `np.full`. In `numpy`, more parameters are supported. Passing a
  number argument as the shape (`np.full(5, value)`) is valid in `numpy` for
  specifying a 1-D shaped result, while TensorFlow does not support this syntax.
  @end_compatibility
  """
  result = gen_array_ops.fill(dims, value, name=name)
  shape_util.maybe_set_static_shape(result, dims)
  return result
