@tf_export("gradients", v1=[])
def gradients_v2(ys,  # pylint: disable=invalid-name
                 xs,
                 grad_ys=None,
                 name="gradients",
                 gate_gradients=False,
                 aggregation_method=None,
                 stop_gradients=None,
                 unconnected_gradients=UnconnectedGradients.NONE):
  """Constructs symbolic derivatives of sum of `ys` w.r.t. x in `xs`.
  `tf.gradients` is only valid in a graph context. In particular,
  it is valid in the context of a `tf.function` wrapper, where code
  is executing as a graph.
  `ys` and `xs` are each a `Tensor` or a list of tensors.  `grad_ys`
  is a list of `Tensor`, holding the gradients received by the
  `ys`. The list must be the same length as `ys`.
  `gradients()` adds ops to the graph to output the derivatives of `ys` with
  respect to `xs`.  It returns a list of `Tensor` of length `len(xs)` where
  each tensor is the `sum(dy/dx)` for y in `ys` and for x in `xs`.
  `grad_ys` is a list of tensors of the same length as `ys` that holds
  the initial gradients for each y in `ys`.  When `grad_ys` is None,
  we fill in a tensor of '1's of the shape of y for each y in `ys`.  A
  user can provide their own initial `grad_ys` to compute the
  derivatives using a different initial gradient for each y (e.g., if
  one wanted to weight the gradient differently for each value in
  each y).
  `stop_gradients` is a `Tensor` or a list of tensors to be considered constant
  with respect to all `xs`. These tensors will not be backpropagated through,
  as though they had been explicitly disconnected using `stop_gradient`.  Among
  other things, this allows computation of partial derivatives as opposed to
  total derivatives. For example:
  >>> @tf.function
  ... def example():
  ...   a = tf.constant(0.)
  ...   b = 2 * a
  ...   return tf.gradients(a + b, [a, b], stop_gradients=[a, b])
  >>> example()
  [<tf.Tensor: shape=(), dtype=float32, numpy=1.0>,
  <tf.Tensor: shape=(), dtype=float32, numpy=1.0>]
  Here the partial derivatives `g` evaluate to `[1.0, 1.0]`, compared to the
  total derivatives `tf.gradients(a + b, [a, b])`, which take into account the
  influence of `a` on `b` and evaluate to `[3.0, 1.0]`.  Note that the above is
  equivalent to:
  >>> @tf.function
  ... def example():
  ...   a = tf.stop_gradient(tf.constant(0.))
  ...   b = tf.stop_gradient(2 * a)
  ...   return tf.gradients(a + b, [a, b])
  >>> example()
  [<tf.Tensor: shape=(), dtype=float32, numpy=1.0>,
  <tf.Tensor: shape=(), dtype=float32, numpy=1.0>]
  `stop_gradients` provides a way of stopping gradient after the graph has
  already been constructed, as compared to `tf.stop_gradient` which is used
  during graph construction.  When the two approaches are combined,
  backpropagation stops at both `tf.stop_gradient` nodes and nodes in
  `stop_gradients`, whichever is encountered first.
  All integer tensors are considered constant with respect to all `xs`, as if
  they were included in `stop_gradients`.
  `unconnected_gradients` determines the value returned for each x in xs if it
  is unconnected in the graph to ys. By default this is None to safeguard
  against errors. Mathematically these gradients are zero which can be requested
  using the `'zero'` option. `tf.UnconnectedGradients` provides the
  following options and behaviors:
  >>> @tf.function
  ... def example(use_zero):
  ...   a = tf.ones([1, 2])
  ...   b = tf.ones([3, 1])
  ...   if use_zero:
  ...     return tf.gradients([b], [a], unconnected_gradients='zero')
  ...   else:
  ...     return tf.gradients([b], [a], unconnected_gradients='none')
  >>> example(False)
  [None]
  >>> example(True)
  [<tf.Tensor: shape=(1, 2), dtype=float32, numpy=array([[0., 0.]], ...)>]
  Let us take one practical example which comes during the back propogation
  phase. This function is used to evaluate the derivatives of the cost function
  with respect to Weights `Ws` and Biases `bs`. Below sample implementation
  provides the exaplantion of what it is actually used for :
  >>> @tf.function
  ... def example():
  ...   Ws = tf.constant(0.)
  ...   bs = 2 * Ws
  ...   cost = Ws + bs  # This is just an example. Please ignore the formulas.
  ...   g = tf.gradients(cost, [Ws, bs])
  ...   dCost_dW, dCost_db = g
  ...   return dCost_dW, dCost_db
  >>> example()
  (<tf.Tensor: shape=(), dtype=float32, numpy=3.0>,
  <tf.Tensor: shape=(), dtype=float32, numpy=1.0>)
  Args:
    ys: A `Tensor` or list of tensors to be differentiated.
    xs: A `Tensor` or list of tensors to be used for differentiation.
    grad_ys: Optional. A `Tensor` or list of tensors the same size as
      `ys` and holding the gradients computed for each y in `ys`.
    name: Optional name to use for grouping all the gradient ops together.
      defaults to 'gradients'.
    gate_gradients: If True, add a tuple around the gradients returned
      for an operations.  This avoids some race conditions.
    aggregation_method: Specifies the method used to combine gradient terms.
      Accepted values are constants defined in the class `AggregationMethod`.
    stop_gradients: Optional. A `Tensor` or list of tensors not to differentiate
      through.
    unconnected_gradients: Optional. Specifies the gradient value returned when
      the given input tensors are unconnected. Accepted values are constants
      defined in the class `tf.UnconnectedGradients` and the default value is
      `none`.
  Returns:
    A list of `Tensor` of length `len(xs)` where each tensor is the `sum(dy/dx)`
    for y in `ys` and for x in `xs`.
  Raises:
    LookupError: if one of the operations between `x` and `y` does not
      have a registered gradient function.
    ValueError: if the arguments are invalid.
    RuntimeError: if called in Eager mode.
  """
  # Creating the gradient graph for control flow mutates Operations.
  # _mutation_lock ensures a Session.run call cannot occur between creating and
  # mutating new ops.
  # pylint: disable=protected-access
  with ops.get_default_graph()._mutation_lock():
    return gradients_util._GradientsHelper(
        ys, xs, grad_ys, name, True, gate_gradients,
        aggregation_method, stop_gradients,
        unconnected_gradients)
  # pylint: enable=protected-access
