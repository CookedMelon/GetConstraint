@tf_export("math.cumsum", "cumsum")
@dispatch.add_dispatch_support
def cumsum(x, axis=0, exclusive=False, reverse=False, name=None):
  """Compute the cumulative sum of the tensor `x` along `axis`.
  By default, this op performs an inclusive cumsum, which means that the first
  element of the input is identical to the first element of the output:
  For example:
  >>> # tf.cumsum([a, b, c])   # [a, a + b, a + b + c]
  >>> x = tf.constant([2, 4, 6, 8])
  >>> tf.cumsum(x)
  <tf.Tensor: shape=(4,), dtype=int32,
  numpy=array([ 2,  6, 12, 20], dtype=int32)>
  >>> # using varying `axis` values
  >>> y = tf.constant([[2, 4, 6, 8], [1,3,5,7]])
  >>> tf.cumsum(y, axis=0)
  <tf.Tensor: shape=(2, 4), dtype=int32, numpy=
  array([[ 2,  4,  6,  8],
         [ 3,  7, 11, 15]], dtype=int32)>
  >>> tf.cumsum(y, axis=1)
  <tf.Tensor: shape=(2, 4), dtype=int32, numpy=
  array([[ 2,  6, 12, 20],
         [ 1,  4,  9, 16]], dtype=int32)>
  By setting the `exclusive` kwarg to `True`, an exclusive cumsum is performed
  instead:
  >>> # tf.cumsum([a, b, c], exclusive=True)  => [0, a, a + b]
  >>> x = tf.constant([2, 4, 6, 8])
  >>> tf.cumsum(x, exclusive=True)
  <tf.Tensor: shape=(4,), dtype=int32,
  numpy=array([ 0,  2,  6, 12], dtype=int32)>
  By setting the `reverse` kwarg to `True`, the cumsum is performed in the
  opposite direction:
  >>> # tf.cumsum([a, b, c], reverse=True)  # [a + b + c, b + c, c]
  >>> x = tf.constant([2, 4, 6, 8])
  >>> tf.cumsum(x, reverse=True)
  <tf.Tensor: shape=(4,), dtype=int32,
  numpy=array([20, 18, 14,  8], dtype=int32)>
  This is more efficient than using separate `tf.reverse` ops.
  The `reverse` and `exclusive` kwargs can also be combined:
  >>> # tf.cumsum([a, b, c], exclusive=True, reverse=True)  # [b + c, c, 0]
  >>> x = tf.constant([2, 4, 6, 8])
  >>> tf.cumsum(x, exclusive=True, reverse=True)
  <tf.Tensor: shape=(4,), dtype=int32,
  numpy=array([18, 14,  8,  0], dtype=int32)>
  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`,
      `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`,
      `complex128`, `qint8`, `quint8`, `qint32`, `half`.
    axis: A `Tensor` of type `int32` (default: 0). Must be in the range
      `[-rank(x), rank(x))`.
    exclusive: If `True`, perform exclusive cumsum.
    reverse: A `bool` (default: False).
    name: A name for the operation (optional).
  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  with ops.name_scope(name, "Cumsum", [x]) as name:
    x = ops.convert_to_tensor(x, name="x")
    return gen_math_ops.cumsum(
        x, axis, exclusive=exclusive, reverse=reverse, name=name)
