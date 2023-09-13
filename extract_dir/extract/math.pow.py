@tf_export("math.pow", "pow")
@dispatch.register_binary_elementwise_api
@dispatch.add_dispatch_support
def pow(x, y, name=None):  # pylint: disable=redefined-builtin
  r"""Computes the power of one value to another.
  Given a tensor `x` and a tensor `y`, this operation computes \\(x^y\\) for
  corresponding elements in `x` and `y`. For example:
  ```python
  x = tf.constant([[2, 2], [3, 3]])
  y = tf.constant([[8, 16], [2, 3]])
  tf.pow(x, y)  # [[256, 65536], [9, 27]]
  ```
  Args:
    x: A `Tensor` of type `float16`, `float32`, `float64`, `int32`, `int64`,
      `complex64`, or `complex128`.
    y: A `Tensor` of type `float16`, `float32`, `float64`, `int32`, `int64`,
      `complex64`, or `complex128`.
    name: A name for the operation (optional).
  Returns:
    A `Tensor`.
  """
  with ops.name_scope(name, "Pow", [x]) as name:
    return gen_math_ops._pow(x, y, name=name)
