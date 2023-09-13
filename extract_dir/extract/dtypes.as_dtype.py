@tf_export("dtypes.as_dtype", "as_dtype")
def as_dtype(type_value):
  """Converts the given `type_value` to a `tf.DType`.
  Inputs can be existing `tf.DType` objects, a [`DataType`
  enum](https://www.tensorflow.org/code/tensorflow/core/framework/types.proto),
  a string type name, or a
  [`numpy.dtype`](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html).
  Examples:
  >>> tf.as_dtype(2)  # Enum value for float64.
  tf.float64
  >>> tf.as_dtype('float')
  tf.float32
  >>> tf.as_dtype(np.int32)
  tf.int32
  Note: `DType` values are interned (i.e. a single instance of each dtype is
  stored in a map). When passed a new `DType` object, `as_dtype` always returns
  the interned value.
  Args:
    type_value: A value that can be converted to a `tf.DType` object.
  Returns:
    A `DType` corresponding to `type_value`.
  Raises:
    TypeError: If `type_value` cannot be converted to a `DType`.
  """
  if isinstance(type_value, DType):
    if type_value._handle_data is None:  # pylint:disable=protected-access
      return _INTERN_TABLE[type_value.as_datatype_enum]
    else:
      return type_value
  if isinstance(type_value, np.dtype):
    try:
      return _NP_TO_TF[type_value.type]
    except KeyError:
      pass
  try:
    return _ANY_TO_TF[type_value]
  except (KeyError, TypeError):
    # TypeError indicates that type_value is not hashable.
    pass
  if hasattr(type_value, "dtype"):
    try:
      return _NP_TO_TF[np.dtype(type_value.dtype).type]
    except (KeyError, TypeError):
      pass
  if isinstance(type_value, _dtypes.DType):
    return _INTERN_TABLE[type_value.as_datatype_enum]
  raise TypeError(f"Cannot convert the argument `type_value`: {type_value!r} "
                  "to a TensorFlow DType.")
