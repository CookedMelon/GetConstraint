@tf_export("math.conj", v1=["math.conj", "conj"])
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints("conj")
def conj(x, name=None):
  r"""Returns the complex conjugate of a complex number.
  Given a tensor `x` of complex numbers, this operation returns a tensor of
  complex numbers that are the complex conjugate of each element in `x`. The
  complex numbers in `x` must be of the form \\(a + bj\\), where `a` is the
  real part and `b` is the imaginary part.
  The complex conjugate returned by this operation is of the form \\(a - bj\\).
  For example:
  >>> x = tf.constant([-2.25 + 4.75j, 3.25 + 5.75j])
  >>> tf.math.conj(x)
  <tf.Tensor: shape=(2,), dtype=complex128,
  numpy=array([-2.25-4.75j,  3.25-5.75j])>
  If `x` is real, it is returned unchanged.
  For example:
  >>> x = tf.constant([-2.25, 3.25])
  >>> tf.math.conj(x)
  <tf.Tensor: shape=(2,), dtype=float32,
  numpy=array([-2.25,  3.25], dtype=float32)>
  Args:
    x: `Tensor` to conjugate.  Must have numeric or variant type.
    name: A name for the operation (optional).
  Returns:
    A `Tensor` that is the conjugate of `x` (with the same type).
  Raises:
    TypeError: If `x` is not a numeric tensor.
  @compatibility(numpy)
  Equivalent to numpy.conj.
  @end_compatibility
  """
  if isinstance(x, ops.Tensor):
    dt = x.dtype
    if dt.is_floating or dt.is_integer:
      return x
  with ops.name_scope(name, "Conj", [x]) as name:
    x = ops.convert_to_tensor(x, name="x")
    if x.dtype.is_complex or x.dtype == dtypes.variant:
      return gen_math_ops.conj(x, name=name)
    elif x.dtype.is_floating or x.dtype.is_integer:
      return x
    else:
      raise TypeError(
          f"Expected numeric or variant tensor, got dtype {x.dtype!r}.")
