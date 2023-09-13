@tf_export("type_spec_from_value")
def type_spec_from_value(value) -> TypeSpec:
  """Returns a `tf.TypeSpec` that represents the given `value`.
  Examples:
    >>> tf.type_spec_from_value(tf.constant([1, 2, 3]))
    TensorSpec(shape=(3,), dtype=tf.int32, name=None)
    >>> tf.type_spec_from_value(np.array([4.0, 5.0], np.float64))
    TensorSpec(shape=(2,), dtype=tf.float64, name=None)
    >>> tf.type_spec_from_value(tf.ragged.constant([[1, 2], [3, 4, 5]]))
    RaggedTensorSpec(TensorShape([2, None]), tf.int32, 1, tf.int64)
    >>> example_input = tf.ragged.constant([[1, 2], [3]])
    >>> @tf.function(input_signature=[tf.type_spec_from_value(example_input)])
    ... def f(x):
    ...   return tf.reduce_sum(x, axis=1)
  Args:
    value: A value that can be accepted or returned by TensorFlow APIs. Accepted
      types for `value` include `tf.Tensor`, any value that can be converted to
      `tf.Tensor` using `tf.convert_to_tensor`, and any subclass of
      `CompositeTensor` (such as `tf.RaggedTensor`).
  Returns:
    A `TypeSpec` that is compatible with `value`.
  Raises:
    TypeError: If a TypeSpec cannot be built for `value`, because its type
      is not supported.
  """
  spec = _type_spec_from_value(value)
  if spec is not None:
    return spec
  # Fallback: try converting value to a tensor.
  try:
    tensor = tensor_conversion_registry.convert(value)
    spec = _type_spec_from_value(tensor)
    if spec is not None:
      return spec
  except (ValueError, TypeError) as e:
    logging.vlog(
        3, "Failed to convert %r to tensor: %s" % (type(value).__name__, e))
  raise TypeError(f"Could not build a TypeSpec for {value} of "
                  f"unsupported type {type(value)}.")
