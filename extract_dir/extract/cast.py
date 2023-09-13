@tf_export("cast", "dtypes.cast")
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def cast(x, dtype, name=None):
  """Casts a tensor to a new type.
  The operation casts `x` (in case of `Tensor`) or `x.values`
  (in case of `SparseTensor` or `IndexedSlices`) to `dtype`.
  For example:
  >>> x = tf.constant([1.8, 2.2], dtype=tf.float32)
  >>> tf.cast(x, tf.int32)
  <tf.Tensor: shape=(2,), dtype=int32, numpy=array([1, 2], dtype=int32)>
  Notice `tf.cast` has an alias `tf.dtypes.cast`:
  >>> x = tf.constant([1.8, 2.2], dtype=tf.float32)
  >>> tf.dtypes.cast(x, tf.int32)
  <tf.Tensor: shape=(2,), dtype=int32, numpy=array([1, 2], dtype=int32)>
  The operation supports data types (for `x` and `dtype`) of
  `uint8`, `uint16`, `uint32`, `uint64`, `int8`, `int16`, `int32`, `int64`,
  `float16`, `float32`, `float64`, `complex64`, `complex128`, `bfloat16`.
  In case of casting from complex types (`complex64`, `complex128`) to real
  types, only the real part of `x` is returned. In case of casting from real
  types to complex types (`complex64`, `complex128`), the imaginary part of the
  returned value is set to `0`. The handling of complex types here matches the
  behavior of numpy.
  Note casting nan and inf values to integral types has undefined behavior.
  Note this operation can lead to a loss of precision when converting native
  Python `float` and `complex` variables to `tf.float64` or `tf.complex128`
  tensors, since the input is first converted to the `float32` data type and
  then widened. It is recommended to use `tf.convert_to_tensor` instead of
  `tf.cast` for any non-tensor inputs.
  Args:
    x: A `Tensor` or `SparseTensor` or `IndexedSlices` of numeric type. It could
      be `uint8`, `uint16`, `uint32`, `uint64`, `int8`, `int16`, `int32`,
      `int64`, `float16`, `float32`, `float64`, `complex64`, `complex128`,
      `bfloat16`.
    dtype: The destination type. The list of supported dtypes is the same as
      `x`.
    name: A name for the operation (optional).
  Returns:
    A `Tensor` or `SparseTensor` or `IndexedSlices` with same shape as `x` and
      same type as `dtype`.
  Raises:
    TypeError: If `x` cannot be cast to the `dtype`.
  """
  base_type = dtypes.as_dtype(dtype).base_dtype
  if isinstance(x,
                (ops.Tensor, _resource_variable_type)) and base_type == x.dtype:
    return x
  with ops.name_scope(name, "Cast", [x]) as name:
    if isinstance(x, sparse_tensor.SparseTensor):
      values_cast = cast(x.values, base_type, name=name)
      x = sparse_tensor.SparseTensor(x.indices, values_cast, x.dense_shape)
    elif isinstance(x, indexed_slices.IndexedSlices):
      values_cast = cast(x.values, base_type, name=name)
      x = indexed_slices.IndexedSlices(values_cast, x.indices, x.dense_shape)
    else:
      # TODO(josh11b): If x is not already a Tensor, we could return
      # ops.convert_to_tensor(x, dtype=dtype, ...)  here, but that
      # allows some conversions that cast() can't do, e.g. casting numbers to
      # strings.
      x = ops.convert_to_tensor(x, name="x")
      if x.dtype.is_complex and base_type.is_floating:
        logging.warn(
            f"You are casting an input of type {x.dtype.name} to an "
            f"incompatible dtype {base_type.name}.  This will "
            "discard the imaginary part and may not be what you "
            "intended."
        )
      if x.dtype != base_type:
        x = gen_math_ops.cast(x, base_type, name=name)
    return x
