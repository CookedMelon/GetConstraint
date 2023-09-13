@tf_export("nn.log_poisson_loss")
@dispatch.add_dispatch_support
def log_poisson_loss(targets, log_input, compute_full_loss=False, name=None):
  """Computes log Poisson loss given `log_input`.
  Gives the log-likelihood loss between the prediction and the target under the
  assumption that the target has a Poisson distribution.
  Caveat: By default, this is not the exact loss, but the loss minus a
    constant term [log(z!)]. That has no effect for optimization, but
    does not play well with relative loss comparisons. To compute an
    approximation of the log factorial term, specify
    compute_full_loss=True to enable Stirling's Approximation.
  For brevity, let `c = log(x) = log_input`, `z = targets`.  The log Poisson
  loss is
        -log(exp(-x) * (x^z) / z!)
      = -log(exp(-x) * (x^z)) + log(z!)
      ~ -log(exp(-x)) - log(x^z) [+ z * log(z) - z + 0.5 * log(2 * pi * z)]
          [ Note the second term is the Stirling's Approximation for log(z!).
            It is invariant to x and does not affect optimization, though
            important for correct relative loss comparisons. It is only
            computed when compute_full_loss == True. ]
      = x - z * log(x) [+ z * log(z) - z + 0.5 * log(2 * pi * z)]
      = exp(c) - z * c [+ z * log(z) - z + 0.5 * log(2 * pi * z)]
  Args:
    targets: A `Tensor` of the same type and shape as `log_input`.
    log_input: A `Tensor` of type `float32` or `float64`.
    compute_full_loss: whether to compute the full loss. If false, a constant
      term is dropped in favor of more efficient optimization.
    name: A name for the operation (optional).
  Returns:
    A `Tensor` of the same shape as `log_input` with the componentwise
    logistic losses.
  Raises:
    ValueError: If `log_input` and `targets` do not have the same shape.
  """
  with ops.name_scope(name, "log_poisson_loss", [log_input, targets]) as name:
    log_input = ops.convert_to_tensor(log_input, name="log_input")
    targets = ops.convert_to_tensor(targets, name="targets")
    try:
      targets.get_shape().assert_is_compatible_with(log_input.get_shape())
    except ValueError:
      raise ValueError(
          "`log_input` and `targets` must have the same shape, received "
          f"({log_input.get_shape()} vs {targets.get_shape()}).")
    result = math_ops.exp(log_input) - log_input * targets
    if compute_full_loss:
      # need to create constant tensors here so that their dtypes can be matched
      # to that of the targets.
      point_five = constant_op.constant(0.5, dtype=targets.dtype)
      two_pi = constant_op.constant(2 * math.pi, dtype=targets.dtype)
      stirling_approx = (targets * math_ops.log(targets)) - targets + (
          point_five * math_ops.log(two_pi * targets))
      zeros = array_ops.zeros_like(targets, dtype=targets.dtype)
      ones = array_ops.ones_like(targets, dtype=targets.dtype)
      cond = math_ops.logical_and(targets >= zeros, targets <= ones)
      result += array_ops.where(cond, zeros, stirling_approx)
    return result
