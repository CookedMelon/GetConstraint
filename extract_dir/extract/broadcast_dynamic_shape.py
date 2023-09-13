@tf_export("broadcast_dynamic_shape")
@dispatch.add_dispatch_support
def broadcast_dynamic_shape(shape_x, shape_y):
  """Computes the shape of a broadcast given symbolic shapes.
  When `shape_x` and `shape_y` are Tensors representing shapes (i.e. the result
  of calling tf.shape on another Tensor) this computes a Tensor which is the
  shape of the result of a broadcasting op applied in tensors of shapes
  `shape_x` and `shape_y`.
  This is useful when validating the result of a broadcasting operation when the
  tensors do not have statically known shapes.
  Example:
  >>> shape_x = (1, 2, 3)
  >>> shape_y = (5, 1, 3)
  >>> tf.broadcast_dynamic_shape(shape_x, shape_y)
  <tf.Tensor: shape=(3,), dtype=int32, numpy=array([5, 2, 3], ...>
  Args:
    shape_x: A rank 1 integer `Tensor`, representing the shape of x.
    shape_y: A rank 1 integer `Tensor`, representing the shape of y.
  Returns:
    A rank 1 integer `Tensor` representing the broadcasted shape.
  Raises:
    InvalidArgumentError: If the two shapes are incompatible for
    broadcasting.
  """
  return gen_array_ops.broadcast_args(shape_x, shape_y)
