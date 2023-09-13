@tf_export("signal.fftshift")
@dispatch.add_dispatch_support
def fftshift(x, axes=None, name=None):
  """Shift the zero-frequency component to the center of the spectrum.
  This function swaps half-spaces for all axes listed (defaults to all).
  Note that ``y[0]`` is the Nyquist component only if ``len(x)`` is even.
  @compatibility(numpy)
  Equivalent to numpy.fft.fftshift.
  https://docs.scipy.org/doc/numpy/reference/generated/numpy.fft.fftshift.html
  @end_compatibility
  For example:
  ```python
  x = tf.signal.fftshift([ 0.,  1.,  2.,  3.,  4., -5., -4., -3., -2., -1.])
  x.numpy() # array([-5., -4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.])
  ```
  Args:
    x: `Tensor`, input tensor.
    axes: `int` or shape `tuple`, optional Axes over which to shift.  Default is
      None, which shifts all axes.
    name: An optional name for the operation.
  Returns:
    A `Tensor`, The shifted tensor.
  """
  with _ops.name_scope(name, "fftshift") as name:
    x = _ops.convert_to_tensor(x)
    if axes is None:
      axes = tuple(range(x.shape.ndims))
      shift = _array_ops.shape(x) // 2
    elif isinstance(axes, int):
      shift = _array_ops.shape(x)[axes] // 2
    else:
      rank = _array_ops.rank(x)
      # allows negative axis
      axes = _array_ops.where(_math_ops.less(axes, 0), axes + rank, axes)
      shift = _array_ops.gather(_array_ops.shape(x), axes) // 2
    return manip_ops.roll(x, shift, axes, name)
