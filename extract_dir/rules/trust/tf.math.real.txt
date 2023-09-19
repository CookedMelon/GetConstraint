@tf_export("math.real", v1=["math.real", "real"])
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints("real")
def real(input, name=None):
  r"""Returns the real part of a complex (or real) tensor.
  Given a tensor `input`, this operation returns a tensor of type `float` that
  is the real part of each element in `input` considered as a complex number.
  For example:
  ```python
  x = tf.constant([-2.25 + 4.75j, 3.25 + 5.75j])
  tf.math.real(x)  # [-2.25, 3.25]
  ```
  If `input` is already real, it is returned unchanged.
  Args:
    input: A `Tensor`. Must have numeric type.
    name: A name for the operation (optional).
  Returns:
    A `Tensor` of type `float32` or `float64`.
  """
  with ops.name_scope(name, "Real", [input]) as name:
    input = ops.convert_to_tensor(input, name="input")
    if input.dtype.is_complex:
      real_dtype = input.dtype.real_dtype
      return gen_math_ops.real(input, Tout=real_dtype, name=name)
    else:
      return input
