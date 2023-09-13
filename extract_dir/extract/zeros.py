@tf_export("zeros")
@dispatch.add_dispatch_support
@_tag_zeros_tensor
def zeros(shape, dtype=dtypes.float32, name=None):
  """Creates a tensor with all elements set to zero.
  See also `tf.zeros_like`, `tf.ones`, `tf.fill`, `tf.eye`.
  This operation returns a tensor of type `dtype` with shape `shape` and
  all elements set to zero.
  >>> tf.zeros([3, 4], tf.int32)
  <tf.Tensor: shape=(3, 4), dtype=int32, numpy=
  array([[0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0]], dtype=int32)>
  Args:
    shape: A `list` of integers, a `tuple` of integers, or
      a 1-D `Tensor` of type `int32`.
    dtype: The DType of an element in the resulting `Tensor`.
    name: Optional string. A name for the operation.
  Returns:
    A `Tensor` with all elements set to zero.
  """
  dtype = dtypes.as_dtype(dtype).base_dtype
  with ops.name_scope(name, "zeros", [shape]) as name:
    if dtype == dtypes.bool:
      zero = False
    elif dtype == dtypes.string:
      zero = ""
    elif dtype.is_quantized:
      zero = np.zeros([]).astype(dtype.as_numpy_dtype)
    else:
      zero = 0
    if not isinstance(shape, ops.Tensor):
      try:
        if not context.executing_eagerly():
          # Create a constant if it won't be very big. Otherwise, create a fill
          # op to prevent serialized GraphDefs from becoming too large.
          output = _constant_if_small(zero, shape, dtype, name)
          if output is not None:
            return output
        # Go through tensor shapes to get int64-if-needed semantics
        shape = constant_op._tensor_shape_tensor_conversion_function(
            tensor_shape.TensorShape(shape))
      except (TypeError, ValueError, errors.UnimplementedError):
        # Happens when shape is a list with tensor elements
        shape = ops.convert_to_tensor(shape, dtype=dtypes.int32)
    if not shape._shape_tuple():
      shape = reshape(shape, [-1])  # Ensure it's a vector
    output = fill(shape, constant(zero, dtype=dtype), name=name)
  assert output.dtype.base_dtype == dtype
  return output
