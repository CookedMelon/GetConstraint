@tf_export("math.divide_no_nan", v1=["math.divide_no_nan", "div_no_nan"])
@dispatch.register_binary_elementwise_api
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints("div_no_nan")
def div_no_nan(x, y, name=None):
  """Computes a safe divide which returns 0 if `y` (denominator) is zero.
  For example:
  >>> tf.constant(3.0) / 0.0
  <tf.Tensor: shape=(), dtype=float32, numpy=inf>
  >>> tf.math.divide_no_nan(3.0, 0.0)
  <tf.Tensor: shape=(), dtype=float32, numpy=0.0>
  Note that 0 is returned if `y` is 0 even if `x` is nonfinite:
  >>> tf.math.divide_no_nan(np.nan, 0.0)
  <tf.Tensor: shape=(), dtype=float32, numpy=0.0>
  Args:
    x: A `Tensor` of a floating or integer dtype.
    y: A `Tensor` with the same dtype as `x` and a compatible shape.
    name: A name for the operation (optional).
  Returns:
    The element-wise quotient as in `tf.math.divide(x, y)`,
    except that division by zero produces `0.0`, not `nan`.
  """
  with ops.name_scope(name, "div_no_nan", [x, y]) as name:
    if not tensor_util.is_tf_type(x) and tensor_util.is_tf_type(y):
      # Treat this case specially like divide() does above.
      y = ops.convert_to_tensor(y, name="y")
      x = ops.convert_to_tensor(x, dtype=y.dtype.base_dtype, name="x")
    else:
      x = ops.convert_to_tensor(x, name="x")
      y = ops.convert_to_tensor(y, dtype_hint=x.dtype.base_dtype, name="y")
    x_dtype = x.dtype.base_dtype
    y_dtype = y.dtype.base_dtype
    if x_dtype != y_dtype:
      raise TypeError(f"`x` and `y` must have the same dtype, "
                      f"got {x_dtype!r} != {y_dtype!r}.")
    try:
      dtype = _TRUEDIV_TABLE[x_dtype]
    except KeyError as e:
      raise TypeError(
          f"Invalid dtype {x_dtype!r} in tf.math.divide_no_nan. Expected one "
          f"of {{{', '.join([repr(x) for x in _TRUEDIV_TABLE.keys()])}}}."
      ) from e
    if dtype is not None:
      x = cast(x, dtype)
      y = cast(y, dtype)
    return gen_math_ops.div_no_nan(x, y, name=name)
