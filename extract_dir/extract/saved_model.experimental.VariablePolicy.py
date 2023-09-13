@tf_export("saved_model.experimental.VariablePolicy")
class VariablePolicy(enum.Enum):
  """Enum defining options for variable handling when saving.
  NONE
    No policy applied: Distributed variables are saved as one variable, with no
    device attached.
  SAVE_VARIABLE_DEVICES
    When saving variables, also save their device assignment.
    This is useful if one wants to hardcode devices in saved models, but it also
    makes them non-portable if soft device placement is disabled (more details
    in `tf.config.set_soft_device_placement`). This is currently not
    fully supported by `saved_model.load`, and is mainly intended to be used
    when one will be reading the saved model at a lower API level. In the
    example below, the graph saved by the call to `saved_model.save` will have
    the variable devices correctly specified:
    ```python
    exported = tf.train.Checkpoint()
    with tf.device('/GPU:0'):
      exported.x_gpu = tf.Variable(1.0)
    with tf.device('/CPU:0'):
      exported.x_cpu = tf.Variable(1.0)
    tf.saved_model.save(exported, export_dir,
        options = tf.saved_model.SaveOptions(
            experimental_variable_policy=
              tf.saved_model.experimental.VariablePolicy.SAVE_VARIABLE_DEVICES))
    ```
    Distributed variables are still saved as one variable under this policy.
  EXPAND_DISTRIBUTED_VARIABLES
    Distributed variables will be saved with information about their components,
    allowing for their restoration on load. Also, the saved graph will contain
    references to those variables. This is useful when one wants to use the
    model for training in environments where the original distribution strategy
    is not available.
  """
  NONE = None
  SAVE_VARIABLE_DEVICES = "save_variable_devices"
  EXPAND_DISTRIBUTED_VARIABLES = "expand_distributed_variables"
  def _save_variable_devices(self):
    """Checks whether variable devices should be saved."""
    return self != VariablePolicy.NONE
  def _expand_distributed_variables(self):
    """Checks whether distributed variables should be expanded."""
    return self == VariablePolicy.EXPAND_DISTRIBUTED_VARIABLES
  @staticmethod
  def from_obj(obj):
    """Tries to convert `obj` to a VariablePolicy instance."""
    if obj is None:
      return VariablePolicy.NONE
    if isinstance(obj, VariablePolicy):
      return obj
    key = str(obj).lower()
    for policy in VariablePolicy:
      if key == policy.value:
        return policy
    raise ValueError(f"Received invalid VariablePolicy value: {obj}.")
