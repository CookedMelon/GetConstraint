@tf_export("nn.weighted_moments", v1=[])
@dispatch.add_dispatch_support
def weighted_moments_v2(x, axes, frequency_weights, keepdims=False, name=None):
  """Returns the frequency-weighted mean and variance of `x`.
  Args:
    x: A tensor.
    axes: 1-d tensor of int32 values; these are the axes along which
      to compute mean and variance.
    frequency_weights: A tensor of positive weights which can be
      broadcast with x.
    keepdims: Produce moments with the same dimensionality as the input.
    name: Name used to scope the operation.
  Returns:
    Two tensors: `weighted_mean` and `weighted_variance`.
  """
  return weighted_moments(
      x=x,
      axes=axes,
      frequency_weights=frequency_weights,
      name=name,
      keep_dims=keepdims)
