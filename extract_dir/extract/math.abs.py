@tf_export("math.abs", "abs")
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def abs(x, name=None):  # pylint: disable=redefined-builtin
  r"""Computes the absolute value of a tensor.
  Given a tensor of integer or floating-point values, this operation returns a
  tensor of the same type, where each element contains the absolute value of the
  corresponding element in the input.
  Given a tensor `x` of complex numbers, this operation returns a tensor of type
  `float32` or `float64` that is the absolute value of each element in `x`. For
  a complex number \\(a + bj\\), its absolute value is computed as
  \\(\sqrt{a^2 + b^2}\\).
  For example:
  >>> # real number
  >>> x = tf.constant([-2.25, 3.25])
  >>> tf.abs(x)
  <tf.Tensor: shape=(2,), dtype=float32,
  numpy=array([2.25, 3.25], dtype=float32)>
  >>> # complex number
  >>> x = tf.constant([[-2.25 + 4.75j], [-3.25 + 5.75j]])
  >>> tf.abs(x)
  <tf.Tensor: shape=(2, 1), dtype=float64, numpy=
  array([[5.25594901],
         [6.60492241]])>
  Args:
    x: A `Tensor` or `SparseTensor` of type `float16`, `float32`, `float64`,
      `int32`, `int64`, `complex64` or `complex128`.
    name: A name for the operation (optional).
  Returns:
    A `Tensor` or `SparseTensor` of the same size, type and sparsity as `x`,
      with absolute values. Note, for `complex64` or `complex128` input, the
      returned `Tensor` will be of type `float32` or `float64`, respectively.
  """
  with ops.name_scope(name, "Abs", [x]) as name:
    x = ops.convert_to_tensor(x, name="x")
    if x.dtype.is_complex:
      return gen_math_ops.complex_abs(x, Tout=x.dtype.real_dtype, name=name)
    return gen_math_ops._abs(x, name=name)
