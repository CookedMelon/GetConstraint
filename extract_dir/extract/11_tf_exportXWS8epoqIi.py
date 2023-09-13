"/home/cc/Workspace/tfconstraint/python/training/experimental/loss_scale.py"
@tf_export(
    v1=[
        'mixed_precision.DynamicLossScale',
        'mixed_precision.experimental.DynamicLossScale',
        'train.experimental.DynamicLossScale'
    ])
class DynamicLossScale(LossScale):
  """Loss scale that dynamically adjusts itself.
  Dynamic loss scaling works by adjusting the loss scale as training progresses.
  The goal is to keep the loss scale as high as possible without overflowing the
  gradients. As long as the gradients do not overflow, raising the loss scale
  never hurts.
  The algorithm starts by setting the loss scale to an initial value. Every N
  steps that the gradients are finite, the loss scale is increased by some
  factor. However, if a NaN or Inf gradient is found, the gradients for that
  step are not applied, and the loss scale is decreased by the factor. This
  process tends to keep the loss scale as high as possible without gradients
  overflowing.
  """
  @deprecation.deprecated(
      None, 'Use tf.keras.mixed_precision.LossScaleOptimizer instead. '
            'LossScaleOptimizer now has all the functionality of '
            'DynamicLossScale')
  def __init__(self,
               initial_loss_scale=2 ** 15,  # See docstring for why this is big.
               increment_period=2000,
               multiplier=2.):
    """Creates the dynamic loss scale.
    Args:
      initial_loss_scale: A Python float.  The loss scale to use at the
        beginning. It's better to start this at a very high number, because a
        loss scale that is too high gets lowered far more quickly than a loss
        scale that is too low gets raised. The default is 2 ** 15, which is
        approximately half the maximum float16 value.
      increment_period: Increases loss scale every `increment_period`
        consecutive steps that finite gradients are encountered. If a nonfinite
        gradient is encountered, the count is reset back to zero.
      multiplier: The multiplier to use when increasing or decreasing the loss
        scale.
    """
    super(DynamicLossScale, self).__init__()
    self._initial_loss_scale = float(initial_loss_scale)
    self._increment_period = int(increment_period)
    self._multiplier = float(multiplier)
    self._current_loss_scale = self._add_weight(
        name='current_loss_scale',
        dtype=dtypes.float32,
        initial_value=self._initial_loss_scale)
    # The number of consecutive steps with finite gradients since the last
    # nonfinite gradient or change in loss scale.
    self._num_good_steps = self._add_weight(
        name='good_steps', dtype=dtypes.int64, initial_value=0)
  @property
  def initial_loss_scale(self):
    return self._initial_loss_scale
  @property
  def increment_period(self):
    return self._increment_period
  @property
  def multiplier(self):
    return self._multiplier
  def __call__(self):
    return ops.convert_to_tensor(self._current_loss_scale)
  def update(self, grads):
    """Updates loss scale based on if gradients are finite in current step."""
    grads = nest.flatten(grads)
    if distribute_lib.has_strategy():
      distribution = distribute_lib.get_cross_replica_context()
      def get_is_finite(grads):
        is_finite = _is_all_finite(grads)
        # We cast to float, because we cannot reduce booleans with
        # DistributionStrategy.
        return math_ops.cast(is_finite, dtypes.float32)
      is_finite_float = distribution.extended.call_for_each_replica(
          get_is_finite, args=(grads,))
      reduced_is_finite_float = distribution.reduce(reduce_util.ReduceOp.SUM,
                                                    is_finite_float, axis=None)
      is_finite = math_ops.equal(reduced_is_finite_float,
                                 distribution.num_replicas_in_sync)
    else:
      is_finite = _is_all_finite(grads)
    def update_if_finite_grads():
      """Update assuming the gradients are finite."""
      def incr_loss_scale():
        new_loss_scale = self._current_loss_scale * self._multiplier
        return control_flow_ops.group(
            _assign_if_finite(self._current_loss_scale, new_loss_scale),
            self._num_good_steps.assign(0))
      return cond.cond(
          self._num_good_steps + 1 >= self._increment_period,
          incr_loss_scale, lambda: _op_in_graph_mode(
              self._num_good_steps.assign_add(1)))
    def update_if_not_finite_grads():
      """Update assuming the gradients are nonfinite."""
      new_loss_scale = math_ops.maximum(
          self._current_loss_scale / self._multiplier, 1)
      return control_flow_ops.group(
          self._num_good_steps.assign(0),
          self._current_loss_scale.assign(new_loss_scale))
    update_op = cond.cond(is_finite, update_if_finite_grads,
                          update_if_not_finite_grads)
    should_apply_gradients = is_finite
    return update_op, should_apply_gradients
  def __repr__(self):
    if context.executing_eagerly():
      return ('DynamicLossScale(current_loss_scale=%s, num_good_steps=%s, '
              'initial_loss_scale=%s, increment_period=%s, multiplier=%s)' %
              (self._current_loss_scale.numpy(), self._num_good_steps.numpy(),
               self.initial_loss_scale, self.increment_period, self.multiplier))
    else:
      return ('DynamicLossScale(initial_loss_scale=%s, increment_period=%s, '
              'multiplier=%s)' %
              (self.initial_loss_scale, self.increment_period, self.multiplier))
  def get_config(self):
    return {
        'initial_loss_scale': self.initial_loss_scale,
        'increment_period': self.increment_period,
        'multiplier': self.multiplier,
    }
def get(identifier):
  """Get a loss scale object."""
  if isinstance(identifier, (int, float)):
    return FixedLossScale(identifier)
  if identifier == 'dynamic':
    return DynamicLossScale()
  if isinstance(identifier, LossScale):
    return identifier
  elif identifier is None:
    return None
  else:
    raise ValueError('Could not interpret loss scale identifier: %s' %
                     identifier)
