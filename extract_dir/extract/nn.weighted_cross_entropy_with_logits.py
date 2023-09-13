@tf_export("nn.weighted_cross_entropy_with_logits", v1=[])
@dispatch.add_dispatch_support
def weighted_cross_entropy_with_logits_v2(labels, logits, pos_weight,
                                          name=None):
  """Computes a weighted cross entropy.
  This is like `sigmoid_cross_entropy_with_logits()` except that `pos_weight`,
  allows one to trade off recall and precision by up- or down-weighting the
  cost of a positive error relative to a negative error.
  The usual cross-entropy cost is defined as:
      labels * -log(sigmoid(logits)) +
          (1 - labels) * -log(1 - sigmoid(logits))
  A value `pos_weight > 1` decreases the false negative count, hence increasing
  the recall.
  Conversely setting `pos_weight < 1` decreases the false positive count and
  increases the precision.
  This can be seen from the fact that `pos_weight` is introduced as a
  multiplicative coefficient for the positive labels term
  in the loss expression:
      labels * -log(sigmoid(logits)) * pos_weight +
          (1 - labels) * -log(1 - sigmoid(logits))
  For brevity, let `x = logits`, `z = labels`, `q = pos_weight`.
  The loss is:
        qz * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
      = qz * -log(1 / (1 + exp(-x))) + (1 - z) * -log(exp(-x) / (1 + exp(-x)))
      = qz * log(1 + exp(-x)) + (1 - z) * (-log(exp(-x)) + log(1 + exp(-x)))
      = qz * log(1 + exp(-x)) + (1 - z) * (x + log(1 + exp(-x))
      = (1 - z) * x + (qz +  1 - z) * log(1 + exp(-x))
      = (1 - z) * x + (1 + (q - 1) * z) * log(1 + exp(-x))
  Setting `l = (1 + (q - 1) * z)`, to ensure stability and avoid overflow,
  the implementation uses
      (1 - z) * x + l * (log(1 + exp(-abs(x))) + max(-x, 0))
  `logits` and `labels` must have the same type and shape.
  >>> labels = tf.constant([1., 0.5, 0.])
  >>> logits = tf.constant([1.5, -0.1, -10.])
  >>> tf.nn.weighted_cross_entropy_with_logits(
  ...     labels=labels, logits=logits, pos_weight=tf.constant(1.5)).numpy()
  array([3.0211994e-01, 8.8049585e-01, 4.5776367e-05], dtype=float32)
  >>> tf.nn.weighted_cross_entropy_with_logits(
  ...     labels=labels, logits=logits, pos_weight=tf.constant(0.5)).numpy()
  array([1.00706644e-01, 5.08297503e-01, 4.57763672e-05], dtype=float32)
  Args:
    labels: A `Tensor` of the same type and shape as `logits`, with values
      between 0 and 1 inclusive.
    logits: A `Tensor` of type `float32` or `float64`, any real numbers.
    pos_weight: A coefficient to use on the positive examples, typically a
      scalar but otherwise broadcastable to the shape of `logits`. Its value
      should be non-negative.
    name: A name for the operation (optional).
  Returns:
    A `Tensor` of the same shape as `logits` with the componentwise
    weighted logistic losses.
  Raises:
    ValueError: If `logits` and `labels` do not have the same shape.
  """
  with ops.name_scope(name, "logistic_loss", [logits, labels]) as name:
    logits = ops.convert_to_tensor(logits, name="logits")
    labels = ops.convert_to_tensor(labels, name="labels")
    try:
      labels.get_shape().assert_is_compatible_with(logits.get_shape())
    except ValueError:
      raise ValueError("`logits` and `labels` must have the same shape, "
                       f"received ({logits.get_shape()} vs "
                       f"{labels.get_shape()}).")
    # The logistic loss formula from above is
    #   (1 - z) * x + (1 + (q - 1) * z) * log(1 + exp(-x))
    # For x < 0, a more numerically stable formula is
    #   (1 - z) * x + (1 + (q - 1) * z) * log(1 + exp(x)) - l * x
    # To avoid branching, we use the combined version
    #   (1 - z) * x + l * (log(1 + exp(-abs(x))) + max(-x, 0))
    log_weight = 1 + (pos_weight - 1) * labels
    return math_ops.add(
        (1 - labels) * logits,
        log_weight * (math_ops.log1p(math_ops.exp(-math_ops.abs(logits))) +
                      nn_ops.relu(-logits)),  # pylint: disable=invalid-unary-operand-type
        name=name)
