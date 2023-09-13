@tf_export("nn.log_softmax", "math.log_softmax", v1=[])
@dispatch.add_dispatch_support
def log_softmax_v2(logits, axis=None, name=None):
  """Computes log softmax activations.
  For each batch `i` and class `j` we have
      logsoftmax = logits - log(reduce_sum(exp(logits), axis))
  Args:
    logits: A non-empty `Tensor`. Must be one of the following types: `half`,
      `float32`, `float64`.
    axis: The dimension softmax would be performed on. The default is -1 which
      indicates the last dimension.
    name: A name for the operation (optional).
  Returns:
    A `Tensor`. Has the same type as `logits`. Same shape as `logits`.
  Raises:
    InvalidArgumentError: if `logits` is empty or `axis` is beyond the last
      dimension of `logits`.
  """
  if axis is None:
    axis = -1
  return _wrap_2d_function(logits, gen_nn_ops.log_softmax, axis, name)
