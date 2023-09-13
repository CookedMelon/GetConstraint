@tf_export("nn.softmax_cross_entropy_with_logits", v1=[])
@dispatch.add_dispatch_support
def softmax_cross_entropy_with_logits_v2(labels, logits, axis=-1, name=None):
  """Computes softmax cross entropy between `logits` and `labels`.
  Measures the probability error in discrete classification tasks in which the
  classes are mutually exclusive (each entry is in exactly one class).  For
  example, each CIFAR-10 image is labeled with one and only one label: an image
  can be a dog or a truck, but not both.
  **NOTE:**  While the classes are mutually exclusive, their probabilities
  need not be.  All that is required is that each row of `labels` is
  a valid probability distribution.  If they are not, the computation of the
  gradient will be incorrect.
  If using exclusive `labels` (wherein one and only
  one class is true at a time), see `sparse_softmax_cross_entropy_with_logits`.
  Usage:
  >>> logits = [[4.0, 2.0, 1.0], [0.0, 5.0, 1.0]]
  >>> labels = [[1.0, 0.0, 0.0], [0.0, 0.8, 0.2]]
  >>> tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
  <tf.Tensor: shape=(2,), dtype=float32,
  numpy=array([0.16984604, 0.82474494], dtype=float32)>
  **WARNING:** This op expects unscaled logits, since it performs a `softmax`
  on `logits` internally for efficiency.  Do not call this op with the
  output of `softmax`, as it will produce incorrect results.
  A common use case is to have logits and labels of shape
  `[batch_size, num_classes]`, but higher dimensions are supported, with
  the `axis` argument specifying the class dimension.
  `logits` and `labels` must have the same dtype (either `float16`, `float32`,
  or `float64`).
  Backpropagation will happen into both `logits` and `labels`.  To disallow
  backpropagation into `labels`, pass label tensors through `tf.stop_gradient`
  before feeding it to this function.
  **Note that to avoid confusion, it is required to pass only named arguments to
  this function.**
  Args:
    labels: Each vector along the class dimension should hold a valid
      probability distribution e.g. for the case in which labels are of shape
      `[batch_size, num_classes]`, each row of `labels[i]` must be a valid
      probability distribution.
    logits: Per-label activations, typically a linear output. These activation
      energies are interpreted as unnormalized log probabilities.
    axis: The class dimension. Defaulted to -1 which is the last dimension.
    name: A name for the operation (optional).
  Returns:
    A `Tensor` that contains the softmax cross entropy loss. Its type is the
    same as `logits` and its shape is the same as `labels` except that it does
    not have the last dimension of `labels`.
  """
  return softmax_cross_entropy_with_logits_v2_helper(
      labels=labels, logits=logits, axis=axis, name=name)
