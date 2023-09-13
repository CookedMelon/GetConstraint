@tf_export("math.polyval")
@dispatch.add_dispatch_support
def polyval(coeffs, x, name=None):
  r"""Computes the elementwise value of a polynomial.
  If `x` is a tensor and `coeffs` is a list n + 1 tensors,
  this function returns the value of the n-th order polynomial
  `p(x) = coeffs[n-1] + coeffs[n-2] * x + ...  + coeffs[0] * x**(n-1)`
  evaluated using Horner's method, i.e.
  ```python
  p(x) = coeffs[n-1] + x * (coeffs[n-2] + ... + x * (coeffs[1] + x * coeffs[0]))
  ```
  Usage Example:
  >>> coefficients = [1.0, 2.5, -4.2]
  >>> x = 5.0
  >>> y = tf.math.polyval(coefficients, x)
  >>> y
  <tf.Tensor: shape=(), dtype=float32, numpy=33.3>
  Usage Example:
  >>> tf.math.polyval([2, 1, 0], 3) # evaluates 2 * (3**2) + 1 * (3**1) + 0 * (3**0)
  <tf.Tensor: shape=(), dtype=int32, numpy=21>
  `tf.math.polyval` can also be used in polynomial regression. Taking
  advantage of this function can facilitate writing a polynomial equation
  as compared to explicitly writing it out, especially for higher degree
  polynomials.
  >>> x = tf.constant(3)
  >>> theta1 = tf.Variable(2)
  >>> theta2 = tf.Variable(1)
  >>> theta3 = tf.Variable(0)
  >>> tf.math.polyval([theta1, theta2, theta3], x)
  <tf.Tensor: shape=(), dtype=int32, numpy=21>
  Args:
    coeffs: A list of `Tensor` representing the coefficients of the polynomial.
    x: A `Tensor` representing the variable of the polynomial.
    name: A name for the operation (optional).
  Returns:
    A `tensor` of the shape as the expression p(x) with usual broadcasting
    rules for element-wise addition and multiplication applied.
  @compatibility(numpy)
  Equivalent to numpy.polyval.
  @end_compatibility
  """
  if not isinstance(coeffs, list):
    raise ValueError(
        f"Argument coeffs must be list type. Received type {type(coeffs)}.")
  with ops.name_scope(name, "polyval", nest.flatten(coeffs) + [x]) as name:
    x = ops.convert_to_tensor(x, name="x")
    if len(coeffs) < 1:
      return array_ops.zeros_like(x, name=name)
    coeffs = [
        ops.convert_to_tensor(coeff, name=("coeff_%d" % index))
        for index, coeff in enumerate(coeffs)
    ]
    p = coeffs[0]
    for c in coeffs[1:]:
      p = c + p * x
    return p
