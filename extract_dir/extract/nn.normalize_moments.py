@tf_export("nn.normalize_moments")
@dispatch.add_dispatch_support
def normalize_moments(counts, mean_ss, variance_ss, shift, name=None):
  """Calculate the mean and variance of based on the sufficient statistics.
  Args:
    counts: A `Tensor` containing the total count of the data (one value).
    mean_ss: A `Tensor` containing the mean sufficient statistics: the (possibly
      shifted) sum of the elements to average over.
    variance_ss: A `Tensor` containing the variance sufficient statistics: the
      (possibly shifted) squared sum of the data to compute the variance over.
    shift: A `Tensor` containing the value by which the data is shifted for
      numerical stability, or `None` if no shift was performed.
    name: Name used to scope the operations that compute the moments.
  Returns:
    Two `Tensor` objects: `mean` and `variance`.
  """
  with ops.name_scope(name, "normalize", [counts, mean_ss, variance_ss, shift]):
    divisor = math_ops.reciprocal(counts, name="divisor")
    if shift is not None:
      shifted_mean = math_ops.multiply(mean_ss, divisor, name="shifted_mean")
      mean = math_ops.add(shifted_mean, shift, name="mean")
    else:  # no shift.
      shifted_mean = math_ops.multiply(mean_ss, divisor, name="mean")
      mean = shifted_mean
    variance = math_ops.subtract(
        math_ops.multiply(variance_ss, divisor),
        math_ops.square(shifted_mean),
        name="variance")
  return (mean, variance)
