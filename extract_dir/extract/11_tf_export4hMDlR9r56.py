"/home/cc/Workspace/tfconstraint/python/training/experimental/loss_scale.py"
@tf_export(
    v1=[
        'mixed_precision.FixedLossScale',
        'mixed_precision.experimental.FixedLossScale',
        'train.experimental.FixedLossScale'
    ])
class FixedLossScale(LossScale):
  """Loss scale with a fixed value.
  The loss scale is not updated for the lifetime of instances of this class.
  A given instance of this class always returns the same number when called.
  """
  @deprecation.deprecated(
      None, 'Use tf.keras.mixed_precision.LossScaleOptimizer instead. '
            'LossScaleOptimizer now has all the functionality of '
            'FixedLossScale')
  def __init__(self, loss_scale_value):
    """Creates the fixed loss scale.
    Args:
      loss_scale_value: A Python float. Its ideal value varies depending on
        models to run. Choosing a too small loss_scale might affect model
        quality; a too big loss_scale might cause inf or nan. There is no single
        right loss_scale to apply. There is no harm choosing a relatively big
        number as long as no nan or inf is encountered in training.
    Raises:
      ValueError: If loss_scale_value is less than 1.
    """
    super(FixedLossScale, self).__init__()
    if not isinstance(loss_scale_value, (int, float)):
      raise ValueError('loss_scale_value must be a Python int or float.')
    if loss_scale_value < 1:
      raise ValueError('loss_scale_value must be at least 1.')
    # It's important we do not create tensors in the constructor, as such
    # tensors might be on a different device or tf.function vs when the tensor
    # is used. This would hurt performance. Therefore, we do not create a tensor
    # from loss_scale_value, but instead leave it as a Python float.
    # TODO(reedwm): Also do not create tensors in the DynamicLossScale
    # constructor.
    self._loss_scale_value = float(loss_scale_value)
  def __call__(self):
    return ops.convert_to_tensor(self._loss_scale_value)
  def update(self, grads):
    del grads
    return control_flow_ops.no_op(), True
  def __repr__(self):
    return 'FixedLossScale(%s)' % self._loss_scale_value
  def get_config(self):
    return {'loss_scale_value': self._loss_scale_value}
def _is_all_finite(grads):
  """Returns a scalar boolean tensor indicating if all gradients are finite."""
  def raw_values(g):
    return g.values if isinstance(g, indexed_slices.IndexedSlices) else g
  is_finite_per_grad = [
      math_ops.reduce_all(math_ops.is_finite(raw_values(g)))
      for g in grads
      if g is not None
  ]
  return math_ops.reduce_all(is_finite_per_grad)
def _op_in_graph_mode(tensor):
  """Returns the tensor's op in graph mode, or the tensor in eager mode.
  This is useful because sometimes an op is needed in graph mode instead of a
  tensor. In eager mode, there are no ops.
  Args:
    tensor: A tensor.
  Returns:
    The tensor's op in graph mode. The tensor in eager mode.
  """
  if context.executing_eagerly():
    return tensor
  return tensor.op
def _assign_if_finite(var, value):
  """Assigns a value to a variable if the value is finite."""
  return cond.cond(
      math_ops.is_finite(value), lambda: _op_in_graph_mode(var.assign(value)),
      control_flow_ops.no_op)
@deprecation.deprecated_endpoints(
    'mixed_precision.experimental.DynamicLossScale',
    'train.experimental.DynamicLossScale')
