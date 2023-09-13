@tf_export("signal.idct", v1=["signal.idct", "spectral.idct"])
@dispatch.add_dispatch_support
def idct(input, type=2, n=None, axis=-1, norm=None, name=None):  # pylint: disable=redefined-builtin
  """Computes the 1D [Inverse Discrete Cosine Transform (DCT)][idct] of `input`.
  Currently Types I, II, III, IV are supported. Type III is the inverse of
  Type II, and vice versa.
  Note that you must re-normalize by 1/(2n) to obtain an inverse if `norm` is
  not `'ortho'`. That is:
  `signal == idct(dct(signal)) * 0.5 / signal.shape[-1]`.
  When `norm='ortho'`, we have:
  `signal == idct(dct(signal, norm='ortho'), norm='ortho')`.
  @compatibility(scipy)
  Equivalent to [scipy.fftpack.idct]
   (https://docs.scipy.org/doc/scipy-1.4.0/reference/generated/scipy.fftpack.idct.html)
   for Type-I, Type-II, Type-III and Type-IV DCT.
  @end_compatibility
  Args:
    input: A `[..., samples]` `float32`/`float64` `Tensor` containing the
      signals to take the DCT of.
    type: The IDCT type to perform. Must be 1, 2, 3 or 4.
    n: For future expansion. The length of the transform. Must be `None`.
    axis: For future expansion. The axis to compute the DCT along. Must be `-1`.
    norm: The normalization to apply. `None` for no normalization or `'ortho'`
      for orthonormal normalization.
    name: An optional name for the operation.
  Returns:
    A `[..., samples]` `float32`/`float64` `Tensor` containing the IDCT of
    `input`.
  Raises:
    ValueError: If `type` is not `1`, `2` or `3`, `n` is not `None, `axis` is
      not `-1`, or `norm` is not `None` or `'ortho'`.
  [idct]:
  https://en.wikipedia.org/wiki/Discrete_cosine_transform#Inverse_transforms
  """
  _validate_dct_arguments(input, type, n, axis, norm)
  inverse_type = {1: 1, 2: 3, 3: 2, 4: 4}[type]
  return _dct_internal(
      input, type=inverse_type, n=n, axis=axis, norm=norm, name=name)
