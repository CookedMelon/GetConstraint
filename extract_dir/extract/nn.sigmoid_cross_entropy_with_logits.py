@tf_export("nn.sigmoid_cross_entropy_with_logits", v1=[])
@dispatch.register_binary_elementwise_api
@dispatch.add_dispatch_support
def sigmoid_cross_entropy_with_logits_v2(  # pylint: disable=invalid-name
    labels=None,
    logits=None,
    name=None):
  r"""Computes sigmoid cross entropy given `logits`.
  Measures the probability error in tasks with two outcomes in which each
  outcome is independent and need not have a fully certain label. For instance,
  one could perform a regression where the probability of an event happening is
  known and used as a label. This loss may also be used for binary
  classification, where labels are either zero or one.
  For brevity, let `x = logits`, `z = labels`.  The logistic loss is
        z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
      = z * -log(1 / (1 + exp(-x))) + (1 - z) * -log(exp(-x) / (1 + exp(-x)))
      = z * log(1 + exp(-x)) + (1 - z) * (-log(exp(-x)) + log(1 + exp(-x)))
      = z * log(1 + exp(-x)) + (1 - z) * (x + log(1 + exp(-x))
      = (1 - z) * x + log(1 + exp(-x))
      = x - x * z + log(1 + exp(-x))
  For x < 0, to avoid overflow in exp(-x), we reformulate the above
        x - x * z + log(1 + exp(-x))
      = log(exp(x)) - x * z + log(1 + exp(-x))
      = - x * z + log(1 + exp(x))
  Hence, to ensure stability and avoid overflow, the implementation uses this
  equivalent formulation
      max(x, 0) - x * z + log(1 + exp(-abs(x)))
  `logits` and `labels` must have the same type and shape.
  >>> logits = tf.constant([1., -1., 0., 1., -1., 0., 0.])
  >>> labels = tf.constant([0., 0., 0., 1., 1., 1., 0.5])
  >>> tf.nn.sigmoid_cross_entropy_with_logits(
  ...     labels=labels, logits=logits).numpy()
  array([1.3132617, 0.3132617, 0.6931472, 0.3132617, 1.3132617, 0.6931472,
         0.6931472], dtype=float32)
  Compared to the losses which handle multiple outcomes,
  `tf.nn.softmax_cross_entropy_with_logits` for general multi-class
  classification and `tf.nn.sparse_softmax_cross_entropy_with_logits` for more
  efficient multi-class classification with hard labels,
  `sigmoid_cross_entropy_with_logits` is a slight simplification for binary
  classification:
        sigmoid(x) = softmax([x, 0])[0]
  $$\frac{1}{1 + e^{-x}} = \frac{e^x}{e^x + e^0}$$
  While `sigmoid_cross_entropy_with_logits` works for soft binary labels
  (probabilities between 0 and 1), it can also be used for binary classification
  where the labels are hard. There is an equivalence between all three symbols
  in this case, with a probability 0 indicating the second class or 1 indicating
  the first class:
  >>> sigmoid_logits = tf.constant([1., -1., 0.])
  >>> softmax_logits = tf.stack([sigmoid_logits, tf.zeros_like(sigmoid_logits)],
  ...                           axis=-1)
  >>> soft_binary_labels = tf.constant([1., 1., 0.])
  >>> soft_multiclass_labels = tf.stack(
  ...     [soft_binary_labels, 1. - soft_binary_labels], axis=-1)
  >>> hard_labels = tf.constant([0, 0, 1])
  >>> tf.nn.sparse_softmax_cross_entropy_with_logits(
  ...     labels=hard_labels, logits=softmax_logits).numpy()
  array([0.31326166, 1.3132616 , 0.6931472 ], dtype=float32)
  >>> tf.nn.softmax_cross_entropy_with_logits(
  ...     labels=soft_multiclass_labels, logits=softmax_logits).numpy()
  array([0.31326166, 1.3132616, 0.6931472], dtype=float32)
  >>> tf.nn.sigmoid_cross_entropy_with_logits(
  ...     labels=soft_binary_labels, logits=sigmoid_logits).numpy()
  array([0.31326166, 1.3132616, 0.6931472], dtype=float32)
  Args:
    labels: A `Tensor` of the same type and shape as `logits`. Between 0 and 1,
      inclusive.
    logits: A `Tensor` of type `float32` or `float64`. Any real number.
    name: A name for the operation (optional).
  Returns:
    A `Tensor` of the same shape as `logits` with the componentwise
    logistic losses.
  Raises:
    ValueError: If `logits` and `labels` do not have the same shape.
  """
  return sigmoid_cross_entropy_with_logits(
      logits=logits, labels=labels, name=name)
