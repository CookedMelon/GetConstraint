@tf_export("quantization.quantize", v1=["quantization.quantize", "quantize"])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints("quantize")
def quantize(
    input,  # pylint: disable=redefined-builtin
    min_range,
    max_range,
    T,
    mode="MIN_COMBINED",
    round_mode="HALF_AWAY_FROM_ZERO",
    name=None,
    narrow_range=False,
    axis=None,
    ensure_minimum_range=0.01):
  """Quantize the input tensor."""
  if ensure_minimum_range != 0.01:
    return quantize_v2(
        input,
        min_range,
        max_range,
        T,
        mode=mode,
        round_mode=round_mode,
        name=name,
        narrow_range=narrow_range,
        axis=axis,
        ensure_minimum_range=ensure_minimum_range)
  return quantize_v2(
      input,
      min_range,
      max_range,
      T,
      mode=mode,
      round_mode=round_mode,
      name=name,
      narrow_range=narrow_range,
      axis=axis)
