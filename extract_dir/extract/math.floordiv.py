@tf_export("math.floordiv", v1=["math.floordiv", "floordiv"])
@dispatch.register_binary_elementwise_api
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints("floordiv")
def floordiv(x, y, name=None):
  """Divides `x / y` elementwise, rounding toward the most negative integer.
  Mathematically, this is equivalent to floor(x / y). For example:
    floor(8.4 / 4.0) = floor(2.1) = 2.0
    floor(-8.4 / 4.0) = floor(-2.1) = -3.0
  This is equivalent to the '//' operator in Python 3.0 and above.
  Note: `x` and `y` must have the same type, and the result will have the same
  type as well.
  Args:
    x: `Tensor` numerator of real numeric type.
    y: `Tensor` denominator of real numeric type.
    name: A name for the operation (optional).
  Returns:
    `x / y` rounded toward -infinity.
  Raises:
    TypeError: If the inputs are complex.
  """
  with ops.name_scope(name, "floordiv", [x, y]) as name:
    return gen_math_ops.floor_div(x, y, name=name)
