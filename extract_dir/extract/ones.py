@tf_export("ones")
@dispatch.add_dispatch_support
def ones(shape, dtype=dtypes.float32, name=None):
  """Creates a tensor with all elements set to one (1).
  See also `tf.ones_like`, `tf.zeros`, `tf.fill`, `tf.eye`.
  This operation returns a tensor of type `dtype` with shape `shape` and
  all elements set to one.
  >>> tf.ones([3, 4], tf.int32)
  <tf.Tensor: shape=(3, 4), dtype=int32, numpy=
  array([[1, 1, 1, 1],
         [1, 1, 1, 1],
         [1, 1, 1, 1]], dtype=int32)>
  Args:
    shape: A `list` of integers, a `tuple` of integers, or
      a 1-D `Tensor` of type `int32`.
    dtype: Optional DType of an element in the resulting `Tensor`. Default is
      `tf.float32`.
    name: Optional string. A name for the operation.
  Returns:
    A `Tensor` with all elements set to one (1).
  """
  dtype = dtypes.as_dtype(dtype).base_dtype
  with ops.name_scope(name, "ones", [shape]) as name:
    if dtype == dtypes.bool:
      one = True
    elif dtype.is_quantized:
      one = np.ones([]).astype(dtype.as_numpy_dtype)
    else:
      one = 1
    if not isinstance(shape, ops.Tensor):
      try:
        if not context.executing_eagerly():
          # Create a constant if it won't be very big. Otherwise, create a fill
          # op to prevent serialized GraphDefs from becoming too large.
          output = _constant_if_small(one, shape, dtype, name)
          if output is not None:
            return output
        # Go through tensor shapes to get int64-if-needed semantics
        shape = constant_op._tensor_shape_tensor_conversion_function(
            tensor_shape.TensorShape(shape))
      except (TypeError, ValueError):
        # Happens when shape is a list with tensor elements
        shape = ops.convert_to_tensor(shape, dtype=dtypes.int32)
    if not shape._shape_tuple():
      shape = reshape(shape, [-1])  # Ensure it's a vector
    output = fill(shape, constant(one, dtype=dtype), name=name)
  assert output.dtype.base_dtype == dtype
  return output
