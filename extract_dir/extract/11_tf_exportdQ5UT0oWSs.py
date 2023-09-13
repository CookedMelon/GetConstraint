"/home/cc/Workspace/tfconstraint/python/training/experimental/loss_scale.py"
@tf_export(
    v1=[
        'mixed_precision.LossScale',
        'mixed_precision.experimental.LossScale',
        'train.experimental.LossScale'
    ])
class LossScale(trackable.Trackable, metaclass=abc.ABCMeta):
  """Base class for all TF1 loss scales.
  This is an abstract base class, so you cannot instantiate it directly.
  Instead, use one of its concrete subclasses:
    * `tf.compat.v1.mixed_precision.DynamicLossScale`
    * `tf.compat.v1.mixed_precision.FixedLossScale`
  Loss scaling is a process that multiplies the loss by a multiplier called the
  loss scale, and divides each gradient by the same multiplier. The pseudocode
  for this process is:
  ```
  loss = ...
  loss *= loss_scale
  grads = gradients(loss, vars)
  grads /= loss_scale
  ```
  Mathematically, loss scaling has no effect, but can help avoid numerical
  underflow in intermediate gradients when float16 tensors are used for mixed
  precision training. By multiplying the loss, each intermediate gradient will
  have the same multiplier applied.
  Instances of this class represent a loss scale. Calling instances of this
  class returns the loss scale as a scalar float32 tensor, while method
  `update()` updates the loss scale depending on the values of the gradients.
  Optimizers use instances of this class to scale loss and gradients.
  In most functions that accept a LossScale, you can also pass an int (such as
  8) to create a `FixedLossScale` or the string `"dynamic"` to create a dynamic
  loss scale.
  """
  def __init__(self):
    """Initializes the loss scale class."""
    self._weights = {}
  @abc.abstractmethod
  def __call__(self):
    """Returns the current loss scale as a scalar `float32` tensor."""
    pass
  @abc.abstractmethod
  def update(self, grads):
    """Updates the value of the loss scale.
    The loss scale will be potentially updated, based on the value of `grads`.
    The tensor returned by calling this class is only updated when this function
    is evaluated.
    In eager mode, this directly updates the loss scale, so that calling
    `__call__` will return the newly updated loss scale. In graph mode,
    this returns an op that, when evaluated, updates the loss scale.
    This function also returns a `should_apply_gradients` bool. If False,
    gradients should not be applied to the variables that step, as nonfinite
    gradients were found, and the loss scale has been be updated to reduce the
    chance of finding nonfinite gradients in the next step. Some loss scale
    classes will always return True, as they cannot adjust themselves in
    response to nonfinite gradients.
    When a DistributionStrategy is used, this function may only be called in a
    cross-replica context.
    Args:
      grads: A nested structure of unscaled gradients, each which is the
        gradient of the loss with respect to a weight. The gradients should have
        already been divided by the loss scale being before passed to this
        function. 'None' gradients are accepted, and are ignored.
    Returns:
      update_op: In eager mode, None. In graph mode, an op to update the loss
        scale.
      should_apply_gradients: Either a bool or a scalar boolean tensor. If
        False, the caller should skip applying `grads` to the variables this
        step.
    """
    pass
  def _add_weight(self, name, initial_value, dtype=None):
    """Adds a weight to this loss scale.
    Args:
      name: Variable name.
      initial_value: The variable's initial value.
      dtype: The type of the variable.
    Returns:
      A variable.
    Raises:
      RuntimeError: If a weight with `name` has already been added.
    """
    variable = variable_v1.VariableV1(
        initial_value=initial_value,
        name=name,
        dtype=dtype,
        trainable=False,
        use_resource=True,
        synchronization=variables.VariableSynchronization.AUTO,
        # Set aggregation to NONE, as loss scaling variables should never be
        # aggregated.
        aggregation=variables.VariableAggregation.NONE)
    if context.executing_eagerly():
      graph_key = None
    else:
      graph = ops.get_default_graph()
      graph_key = graph._graph_key  # pylint: disable=protected-access
    key = (name, graph_key)
    if self._weights.get(key, None) is not None:
      raise RuntimeError('Duplicate variables detected. {}'.format(key))
    self._weights[key] = variable
    self._handle_deferred_dependencies(name=name, trackable=variable)
    return variable
  def _trackable_children(self,
                          save_type=trackable.SaveType.CHECKPOINT,
                          **kwargs):
    """From Trackable. Gather graph-specific weights to save."""
    if context.executing_eagerly():
      graph_key = None
    else:
      graph = ops.get_default_graph()
      graph_key = graph._graph_key  # pylint: disable=protected-access
    weights = {}
    for (name, g), v in sorted(self._weights.items(), key=lambda i: i[0][0]):
      if g == graph_key:
        weights[name] = v
    weights.update(
        super(LossScale, self)._trackable_children(save_type, **kwargs))
    return weights
  def _lookup_dependency(self, name):
    """From Trackable. Find a weight in the current graph."""
    unconditional = super(LossScale, self)._lookup_dependency(name)
    if unconditional is not None:
      return unconditional
    if context.executing_eagerly():
      graph_key = None
    else:
      graph = ops.get_default_graph()
      graph_key = graph._graph_key  # pylint: disable=protected-access
    return self._weights.get((name, graph_key), None)
  @abc.abstractmethod
  def get_config(self):
    """Returns the config of this loss scale."""
    pass
  @classmethod
  def from_config(cls, config):
    """Creates the LossScale from its config."""
    return cls(**config)
@deprecation.deprecated_endpoints('mixed_precision.experimental.FixedLossScale',
                                  'train.experimental.FixedLossScale')
