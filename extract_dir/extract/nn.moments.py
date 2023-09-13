@tf_export("nn.moments", v1=[])
@dispatch.add_dispatch_support
def moments_v2(
    x,
    axes,
    shift=None,
    keepdims=False,
    name=None):
  """Calculates the mean and variance of `x`.
  The mean and variance are calculated by aggregating the contents of `x`
  across `axes`.  If `x` is 1-D and `axes = [0]` this is just the mean
  and variance of a vector.
  Note: shift is currently not used; the true mean is computed and used.
  When using these moments for batch normalization (see
  `tf.nn.batch_normalization`):
   * for so-called "global normalization", used with convolutional filters with
     shape `[batch, height, width, depth]`, pass `axes=[0, 1, 2]`.
   * for simple batch normalization pass `axes=[0]` (batch only).
  Args:
    x: A `Tensor`.
    axes: Array of ints.  Axes along which to compute mean and
      variance.
    shift: Not used in the current implementation.
    keepdims: produce moments with the same dimensionality as the input.
    name: Name used to scope the operations that compute the moments.
  Returns:
    Two `Tensor` objects: `mean` and `variance`.
  """
  return moments(x=x, axes=axes, shift=shift, name=name, keep_dims=keepdims)
