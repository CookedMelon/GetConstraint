@tf_export("math.multiply_no_nan")
@dispatch.register_binary_elementwise_api
@dispatch.add_dispatch_support
def multiply_no_nan(x, y, name=None):
  """Computes the product of x and y and returns 0 if the y is zero, even if x is NaN or infinite.
  Note this is noncommutative: if y is NaN or infinite and x is 0, the result
  will be NaN.
  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    y: A `Tensor` whose dtype is compatible with `x`.
    name: A name for the operation (optional).
  Returns:
    The element-wise value of the x times y.
  """
  with ops.name_scope(name, "multiply_no_nan", [x, y]) as name:
    x = ops.convert_to_tensor(x, name="x")
    y = ops.convert_to_tensor(y, name="y", dtype=x.dtype.base_dtype)
    x_dtype = x.dtype.base_dtype
    y_dtype = y.dtype.base_dtype
    if x_dtype != y_dtype:
      raise TypeError(f"`x` and `y` must have the same dtype, "
                      f"got {x_dtype!r} != {y_dtype!r}")
    return gen_math_ops.mul_no_nan(x, y, name=name)
