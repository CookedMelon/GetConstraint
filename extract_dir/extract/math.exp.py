@tf_export("math.exp", "exp")
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def exp(x, name=None):
  r"""Computes exponential of x element-wise.  \\(y = e^x\\).
  This function computes the exponential of the input tensor element-wise.
  i.e. `math.exp(x)` or \\(e^x\\), where `x` is the input tensor.
  \\(e\\) denotes Euler's number and is approximately equal to 2.718281.
  Output is positive for any real input.
  >>> x = tf.constant(2.0)
  >>> tf.math.exp(x)
  <tf.Tensor: shape=(), dtype=float32, numpy=7.389056>
  >>> x = tf.constant([2.0, 8.0])
  >>> tf.math.exp(x)
  <tf.Tensor: shape=(2,), dtype=float32,
  numpy=array([   7.389056, 2980.958   ], dtype=float32)>
  For complex numbers, the exponential value is calculated as
  $$
  e^{x+iy} = {e^x} {e^{iy}} = {e^x} ({\cos (y) + i \sin (y)})
  $$
  For `1+1j` the value would be computed as:
  $$
  e^1 (\cos (1) + i \sin (1)) = 2.7182817 \times (0.5403023+0.84147096j)
  $$
  >>> x = tf.constant(1 + 1j)
  >>> tf.math.exp(x)
  <tf.Tensor: shape=(), dtype=complex128,
  numpy=(1.4686939399158851+2.2873552871788423j)>
  Args:
    x: A `tf.Tensor`. Must be one of the following types: `bfloat16`, `half`,
      `float32`, `float64`, `complex64`, `complex128`.
    name: A name for the operation (optional).
  Returns:
    A `tf.Tensor`. Has the same type as `x`.
  @compatibility(numpy)
  Equivalent to np.exp
  @end_compatibility
  """
  return gen_math_ops.exp(x, name)
