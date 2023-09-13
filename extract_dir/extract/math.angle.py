@tf_export("math.angle", v1=["math.angle", "angle"])
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints("angle")
def angle(input, name=None):
  r"""Returns the element-wise argument of a complex (or real) tensor.
  Given a tensor `input`, this operation returns a tensor of type `float` that
  is the argument of each element in `input` considered as a complex number.
  The elements in `input` are considered to be complex numbers of the form
  \\(a + bj\\), where *a* is the real part and *b* is the imaginary part.
  If `input` is real then *b* is zero by definition.
  The argument returned by this function is of the form \\(atan2(b, a)\\).
  If `input` is real, a tensor of all zeros is returned.
  For example:
  ```
  input = tf.constant([-2.25 + 4.75j, 3.25 + 5.75j], dtype=tf.complex64)
  tf.math.angle(input).numpy()
  # ==> array([2.0131705, 1.056345 ], dtype=float32)
  ```
  Args:
    input: A `Tensor`. Must be one of the following types: `float`, `double`,
      `complex64`, `complex128`.
    name: A name for the operation (optional).
  Returns:
    A `Tensor` of type `float32` or `float64`.
  """
  with ops.name_scope(name, "Angle", [input]) as name:
    input = ops.convert_to_tensor(input, name="input")
    if input.dtype.is_complex:
      return gen_math_ops.angle(input, Tout=input.dtype.real_dtype, name=name)
    else:
      return array_ops.where(input < 0, np.pi * array_ops.ones_like(input),
                             array_ops.zeros_like(input))
