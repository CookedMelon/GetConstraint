@tf_export("math.sign", "sign")
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def sign(x, name=None):
  r"""Returns an element-wise indication of the sign of a number.
  `y = sign(x) = -1 if x < 0; 0 if x == 0; 1 if x > 0`.
  For complex numbers, `y = sign(x) = x / |x| if x != 0, otherwise y = 0`.
  Example usage:
  >>> # real number
  >>> tf.math.sign([0., 2., -3.])
  <tf.Tensor: shape=(3,), dtype=float32,
  numpy=array([ 0.,  1., -1.], dtype=float32)>
  >>> # complex number
  >>> tf.math.sign([1 + 1j, 0 + 0j])
  <tf.Tensor: shape=(2,), dtype=complex128,
  numpy=array([0.70710678+0.70710678j, 0.        +0.j        ])>
  Args:
   x: A Tensor. Must be one of the following types: bfloat16, half, float32,
     float64, int32, int64, complex64, complex128.
   name: A name for the operation (optional).
  Returns:
   A Tensor. Has the same type as x.
   If x is a SparseTensor, returns SparseTensor(x.indices,
     tf.math.sign(x.values, ...), x.dense_shape).
  """
  x = ops.convert_to_tensor(x)
  if x.dtype.is_complex:
    return gen_math_ops.div_no_nan(
        x,
        cast(
            gen_math_ops.complex_abs(
                x,
                Tout=dtypes.float32
                if x.dtype == dtypes.complex64 else dtypes.float64),
            dtype=x.dtype),
        name=name)
  return gen_math_ops.sign(x, name=name)
