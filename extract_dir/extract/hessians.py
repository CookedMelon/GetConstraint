@tf_export("hessians", v1=[])
def HessiansV2(ys,
               xs,
               gate_gradients=False,
               aggregation_method=None,
               name="hessians"):
  """Constructs the Hessian of sum of `ys` with respect to `x` in `xs`.
  `hessians()` adds ops to the graph to output the Hessian matrix of `ys`
  with respect to `xs`.  It returns a list of `Tensor` of length `len(xs)`
  where each tensor is the Hessian of `sum(ys)`.
  The Hessian is a matrix of second-order partial derivatives of a scalar
  tensor (see https://en.wikipedia.org/wiki/Hessian_matrix for more details).
  Args:
    ys: A `Tensor` or list of tensors to be differentiated.
    xs: A `Tensor` or list of tensors to be used for differentiation.
    gate_gradients: See `gradients()` documentation for details.
    aggregation_method: See `gradients()` documentation for details.
    name: Optional name to use for grouping all the gradient ops together.
      defaults to 'hessians'.
  Returns:
    A list of Hessian matrices of `sum(ys)` for each `x` in `xs`.
  Raises:
    LookupError: if one of the operations between `xs` and `ys` does not
      have a registered gradient function.
  """
  return hessians(
      ys,
      xs,
      name=name,
      colocate_gradients_with_ops=True,
      gate_gradients=gate_gradients,
      aggregation_method=aggregation_method)
