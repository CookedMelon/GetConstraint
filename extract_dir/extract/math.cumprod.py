@tf_export("math.cumprod", v1=["math.cumprod", "cumprod"])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints("cumprod")
def cumprod(x, axis=0, exclusive=False, reverse=False, name=None):
  """Compute the cumulative product of the tensor `x` along `axis`.
  By default, this op performs an inclusive cumprod, which means that the
  first element of the input is identical to the first element of the output:
  ```python
  tf.math.cumprod([a, b, c])  # [a, a * b, a * b * c]
  ```
  By setting the `exclusive` kwarg to `True`, an exclusive cumprod is
  performed
  instead:
  ```python
  tf.math.cumprod([a, b, c], exclusive=True)  # [1, a, a * b]
  ```
  By setting the `reverse` kwarg to `True`, the cumprod is performed in the
  opposite direction:
  ```python
  tf.math.cumprod([a, b, c], reverse=True)  # [a * b * c, b * c, c]
  ```
  This is more efficient than using separate `tf.reverse` ops.
  The `reverse` and `exclusive` kwargs can also be combined:
  ```python
  tf.math.cumprod([a, b, c], exclusive=True, reverse=True)  # [b * c, c, 1]
  ```
  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`,
      `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`,
      `complex128`, `qint8`, `quint8`, `qint32`, `half`.
    axis: A `Tensor` of type `int32` (default: 0). Must be in the range
      `[-rank(x), rank(x))`.
    exclusive: If `True`, perform exclusive cumprod.
    reverse: A `bool` (default: False).
    name: A name for the operation (optional).
  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  with ops.name_scope(name, "Cumprod", [x]) as name:
    x = ops.convert_to_tensor(x, name="x")
    return gen_math_ops.cumprod(
        x, axis, exclusive=exclusive, reverse=reverse, name=name)
