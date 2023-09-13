@tf_export("math.argmin", "argmin", v1=[])
@dispatch.add_dispatch_support
def argmin_v2(input, axis=None, output_type=dtypes.int64, name=None):
  """Returns the index with the smallest value across axes of a tensor.
  Returns the smallest index in case of ties.
  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`,
      `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`,
      `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`,
      `uint64`.
    axis: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      int32 or int64, must be in the range `-rank(input), rank(input))`.
      Describes which axis of the input Tensor to reduce across. For vectors,
      use axis = 0.
    output_type: An optional `tf.DType` from: `tf.int32, tf.int64`. Defaults to
      `tf.int64`.
    name: A name for the operation (optional).
  Returns:
    A `Tensor` of type `output_type`.
  Usage:
  ```python
  import tensorflow as tf
  a = [1, 10, 26.9, 2.8, 166.32, 62.3]
  b = tf.math.argmin(input = a)
  c = tf.keras.backend.eval(b)
  # c = 0
  # here a[0] = 1 which is the smallest element of a across axis 0
  ```
  """
  if axis is None:
    axis = 0
  return gen_math_ops.arg_min(input, axis, name=name, output_type=output_type)
