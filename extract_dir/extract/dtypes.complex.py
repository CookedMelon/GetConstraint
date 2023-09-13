@tf_export("dtypes.complex", "complex")
@dispatch.register_binary_elementwise_api
@dispatch.add_dispatch_support
def complex(real, imag, name=None):
  r"""Converts two real numbers to a complex number.
  Given a tensor `real` representing the real part of a complex number, and a
  tensor `imag` representing the imaginary part of a complex number, this
  operation returns complex numbers elementwise of the form \\(a + bj\\), where
  *a* represents the `real` part and *b* represents the `imag` part.
  The input tensors `real` and `imag` must have the same shape.
  For example:
  ```python
  real = tf.constant([2.25, 3.25])
  imag = tf.constant([4.75, 5.75])
  tf.complex(real, imag)  # [[2.25 + 4.75j], [3.25 + 5.75j]]
  ```
  Args:
    real: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    imag: A `Tensor`. Must have the same type as `real`.
    name: A name for the operation (optional).
  Returns:
    A `Tensor` of type `complex64` or `complex128`.
  Raises:
    TypeError: Real and imag must be correct types
  """
  real = ops.convert_to_tensor(real, name="real")
  imag = ops.convert_to_tensor(imag, name="imag")
  with ops.name_scope(name, "Complex", [real, imag]) as name:
    input_types = (real.dtype, imag.dtype)
    if input_types == (dtypes.float64, dtypes.float64):
      Tout = dtypes.complex128
    elif input_types == (dtypes.float32, dtypes.float32):
      Tout = dtypes.complex64
    else:
      raise TypeError(
          f"The `real` and `imag` components have incorrect types: "
          f"{real.dtype.name} {imag.dtype.name}. They must be consistent, and "
          f"one of {[dtypes.float32, dtypes.float64]}")
    return gen_math_ops._complex(real, imag, Tout=Tout, name=name)
