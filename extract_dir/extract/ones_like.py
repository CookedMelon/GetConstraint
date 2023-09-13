@tf_export("ones_like", v1=[])
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def ones_like_v2(
    input,  # pylint: disable=redefined-builtin
    dtype=None,
    name=None):
  """Creates a tensor of all ones that has the same shape as the input.
  See also `tf.ones`.
  Given a single tensor (`tensor`), this operation returns a tensor of the
  same type and shape as `tensor` with all elements set to 1. Optionally,
  you can use `dtype` to specify a new type for the returned tensor.
  For example:
  >>> tensor = tf.constant([[1, 2, 3], [4, 5, 6]])
  >>> tf.ones_like(tensor)
  <tf.Tensor: shape=(2, 3), dtype=int32, numpy=
    array([[1, 1, 1],
           [1, 1, 1]], dtype=int32)>
  Args:
    input: A `Tensor`.
    dtype: A type for the returned `Tensor`. Must be `float16`, `float32`,
      `float64`, `int8`, `uint8`, `int16`, `uint16`, `int32`, `int64`,
      `complex64`, `complex128`, `bool` or `string`.
    name: A name for the operation (optional).
  Returns:
    A `Tensor` with all elements set to one.
  """
  return ones_like_impl(input, dtype, name, optimize=True)
