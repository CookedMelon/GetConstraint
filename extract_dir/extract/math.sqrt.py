@tf_export("math.sqrt", "sqrt")
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def sqrt(x, name=None):  # pylint: disable=redefined-builtin
  r"""Computes element-wise square root of the input tensor.
  Note: This operation does not support integer types.
  >>> x = tf.constant([[4.0], [16.0]])
  >>> tf.sqrt(x)
  <tf.Tensor: shape=(2, 1), dtype=float32, numpy=
    array([[2.],
           [4.]], dtype=float32)>
  >>> y = tf.constant([[-4.0], [16.0]])
  >>> tf.sqrt(y)
  <tf.Tensor: shape=(2, 1), dtype=float32, numpy=
    array([[nan],
           [ 4.]], dtype=float32)>
  >>> z = tf.constant([[-1.0], [16.0]], dtype=tf.complex128)
  >>> tf.sqrt(z)
  <tf.Tensor: shape=(2, 1), dtype=complex128, numpy=
    array([[0.0+1.j],
           [4.0+0.j]])>
  Note: In order to support complex type, please provide an input tensor
  of `complex64` or `complex128`.
  Args:
    x: A `tf.Tensor` of type `bfloat16`, `half`, `float32`, `float64`,
      `complex64`, `complex128`
    name: A name for the operation (optional).
  Returns:
    A `tf.Tensor` of same size, type and sparsity as `x`.
  """
  return gen_math_ops.sqrt(x, name)
