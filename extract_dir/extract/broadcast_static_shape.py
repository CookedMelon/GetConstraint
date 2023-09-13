@tf_export("broadcast_static_shape")
@dispatch.add_dispatch_support
def broadcast_static_shape(shape_x, shape_y):
  """Computes the shape of a broadcast given known shapes.
  When `shape_x` and `shape_y` are fully known `TensorShape`s this computes a
  `TensorShape` which is the shape of the result of a broadcasting op applied in
  tensors of shapes `shape_x` and `shape_y`.
  For example, if shape_x is `TensorShape([1, 2, 3])` and shape_y is
  `TensorShape([5, 1, 3])`, the result is a TensorShape whose value is
  `TensorShape([5, 2, 3])`.
  This is useful when validating the result of a broadcasting operation when the
  tensors have statically known shapes.
  Example:
  >>> shape_x = tf.TensorShape([1, 2, 3])
  >>> shape_y = tf.TensorShape([5, 1 ,3])
  >>> tf.broadcast_static_shape(shape_x, shape_y)
  TensorShape([5, 2, 3])
  Args:
    shape_x: A `TensorShape`
    shape_y: A `TensorShape`
  Returns:
    A `TensorShape` representing the broadcasted shape.
  Raises:
    ValueError: If the two shapes can not be broadcasted.
  """
  return common_shapes.broadcast_shape(shape_x, shape_y)
