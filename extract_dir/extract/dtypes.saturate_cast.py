@tf_export("dtypes.saturate_cast", "saturate_cast")
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def saturate_cast(value, dtype, name=None):
  """Performs a safe saturating cast of `value` to `dtype`.
  This function casts the input to `dtype` without overflow.  If
  there is a danger that values would over or underflow in the cast, this op
  applies the appropriate clamping before the cast.  See `tf.cast` for more
  details.
  Args:
    value: A `Tensor`.
    dtype: The desired output `DType`.
    name: A name for the operation (optional).
  Returns:
    `value` safely cast to `dtype`.
  """
  # When casting to a type with smaller representable range, clamp.
  # Note that this covers casting to unsigned types as well.
  with ops.name_scope(name, "saturate_cast", [value]) as name:
    value = ops.convert_to_tensor(value, name="value")
    dtype = dtypes.as_dtype(dtype).base_dtype
    in_dtype = value.dtype
    if in_dtype.is_complex:
      if dtype.is_complex:
        # Clamp real and imag components separately, if required.
        real_in_dtype = in_dtype.real_dtype
        real_out_dtype = dtype.real_dtype
        if real_in_dtype.min < real_out_dtype.min or real_in_dtype.max > real_out_dtype.max:
          value = gen_math_ops._clip_by_value(
              value,
              ops.convert_to_tensor(
                  builtins.complex(real_out_dtype.min, real_out_dtype.min),
                  dtype=in_dtype),
              ops.convert_to_tensor(
                  builtins.complex(real_out_dtype.max, real_out_dtype.max),
                  dtype=in_dtype),
              name="clamp")
        return cast(value, dtype, name=name)
      else:
        # Extract real component and fall through to clamp+cast.
        value = real(value)
        logging.warn("Casting complex to real discards imaginary part.")
        in_dtype = in_dtype.real_dtype
    # in_dtype is real, but out_dtype could be complex.
    out_real_dtype = dtype.real_dtype
    if in_dtype.min < out_real_dtype.min or in_dtype.max > out_real_dtype.max:
      # The output min/max may not actually be representable in the
      # in_dtype (e.g. casting float32 to uint32).  This can lead to undefined
      # behavior when trying to cast a value outside the valid range of the
      # target type. We work around this by nudging the min/max to fall within
      # the valid output range.  The catch is that we may actually saturate
      # to a value less than the true saturation limit, but this is the best we
      # can do in order to avoid UB without introducing a separate SaturateCast
      # op.
      min_limit = in_dtype.as_numpy_dtype(out_real_dtype.min)
      if min_limit < out_real_dtype.min:
        min_limit = np.nextafter(
            out_real_dtype.min, 0, dtype=in_dtype.as_numpy_dtype
        )
      max_limit = in_dtype.as_numpy_dtype(out_real_dtype.max)
      if max_limit > out_real_dtype.max:
        max_limit = np.nextafter(
            out_real_dtype.max, 0, dtype=in_dtype.as_numpy_dtype
        )
      value = gen_math_ops._clip_by_value(
          value,
          ops.convert_to_tensor(min_limit, dtype=in_dtype),
          ops.convert_to_tensor(max_limit, dtype=in_dtype),
          name="clamp",
      )
    return cast(value, dtype, name=name)
