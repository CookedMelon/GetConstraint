@tf_export("zeros_like", v1=[])
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def zeros_like_v2(
    input,  # pylint: disable=redefined-builtin
    dtype=None,
    name=None):
  """Creates a tensor with all elements set to zero.
  See also `tf.zeros`.
  Given a single tensor or array-like object (`input`), this operation returns
  a tensor of the same type and shape as `input` with all elements set to zero.
  Optionally, you can use `dtype` to specify a new type for the returned tensor.
  Examples:
    >>> tensor = tf.constant([[1, 2, 3], [4, 5, 6]])
    >>> tf.zeros_like(tensor)
    <tf.Tensor: shape=(2, 3), dtype=int32, numpy=
    array([[0, 0, 0],
           [0, 0, 0]], dtype=int32)>
    >>> tf.zeros_like(tensor, dtype=tf.float32)
    <tf.Tensor: shape=(2, 3), dtype=float32, numpy=
    array([[0., 0., 0.],
           [0., 0., 0.]], dtype=float32)>
    >>> tf.zeros_like([[1, 2, 3], [4, 5, 6]])
    <tf.Tensor: shape=(2, 3), dtype=int32, numpy=
    array([[0, 0, 0],
           [0, 0, 0]], dtype=int32)>
  Args:
    input: A `Tensor` or array-like object.
    dtype: A type for the returned `Tensor`. Must be `float16`, `float32`,
      `float64`, `int8`, `uint8`, `int16`, `uint16`, `int32`, `int64`,
      `complex64`, `complex128`, `bool` or `string` (optional).
    name: A name for the operation (optional).
  Returns:
    A `Tensor` with all elements set to zero.
  """
  return zeros_like_impl(input, dtype, name, optimize=True)
