@tf_export("tpu.experimental.embedding.Adagrad")
class Adagrad(_Optimizer):
  """Optimization parameters for Adagrad with TPU embeddings.
  Pass this to `tf.tpu.experimental.embedding.TPUEmbedding` via the `optimizer`
  argument to set the global optimizer and its parameters:
  ```python
  embedding = tf.tpu.experimental.embedding.TPUEmbedding(
      ...
      optimizer=tf.tpu.experimental.embedding.Adagrad(0.1))
  ```
  This can also be used in a `tf.tpu.experimental.embedding.TableConfig` as the
  optimizer parameter to set a table specific optimizer. This will override the
  optimizer and parameters for global embedding optimizer defined above:
  ```python
  table_one = tf.tpu.experimental.embedding.TableConfig(
      vocabulary_size=...,
      dim=...,
      optimizer=tf.tpu.experimental.embedding.Adagrad(0.2))
  table_two = tf.tpu.experimental.embedding.TableConfig(
      vocabulary_size=...,
      dim=...)
  feature_config = (
      tf.tpu.experimental.embedding.FeatureConfig(
          table=table_one),
      tf.tpu.experimental.embedding.FeatureConfig(
          table=table_two))
  embedding = tf.tpu.experimental.embedding.TPUEmbedding(
      feature_config=feature_config,
      batch_size=...
      optimizer=tf.tpu.experimental.embedding.Adagrad(0.1))
  ```
  In the above example, the first feature will be looked up in a table that has
  a learning rate of 0.2 while the second feature will be looked up in a table
  that has a learning rate of 0.1.
  See 'tensorflow/core/protobuf/tpu/optimization_parameters.proto' for a
  complete description of these parameters and their impacts on the optimizer
  algorithm.
  """
  def __init__(
      self,
      learning_rate: Union[float, Callable[[], float]] = 0.001,
      initial_accumulator_value: float = 0.1,
      use_gradient_accumulation: bool = True,
      clip_weight_min: Optional[float] = None,
      clip_weight_max: Optional[float] = None,
      weight_decay_factor: Optional[float] = None,
      multiply_weight_decay_factor_by_learning_rate: bool = None,
      slot_variable_creation_fn: Optional[SlotVarCreationFnType] = None,
      clipvalue: Optional[ClipValueType] = None,
      low_dimensional_packing_status: bool = False,
  ):
    """Optimization parameters for Adagrad.
    Args:
      learning_rate: The learning rate. It should be a floating point value or a
        callable taking no arguments for a dynamic learning rate.
      initial_accumulator_value: initial accumulator for Adagrad.
      use_gradient_accumulation: setting this to `False` makes embedding
        gradients calculation less accurate but faster.
      clip_weight_min: the minimum value to clip by; None means -infinity.
      clip_weight_max: the maximum value to clip by; None means +infinity.
      weight_decay_factor: amount of weight decay to apply; None means that the
        weights are not decayed.
      multiply_weight_decay_factor_by_learning_rate: if true,
        `weight_decay_factor` is multiplied by the current learning rate.
      slot_variable_creation_fn: If you wish do directly control the creation of
        the slot variables, set this to a callable taking three parameters: a
        table variable, a list of slot names to create for it, and a list of
        initializers. This function should return a dict with the slot names as
        keys and the created variables as values with types matching the table
        variable. When set to None (the default), uses the built-in variable
        creation.
      clipvalue: Controls clipping of the gradient. Set to either a single
        positive scalar value to get clipping or a tuple of scalar values (min,
        max) to set a separate maximum or minimum. If one of the two entries is
        None, then there will be no clipping that direction.
      low_dimensional_packing_status: Status of the low-dimensional embedding
        packing optimization controls whether to optimize the packing of
        1-dimensional, 2-dimensional, and 4-dimensional embedding tables in
        memory.
    """
    super().__init__(
        learning_rate,
        use_gradient_accumulation,
        clip_weight_min,
        clip_weight_max,
        weight_decay_factor,
        multiply_weight_decay_factor_by_learning_rate,
        clipvalue,
        slot_variable_creation_fn,
        low_dimensional_packing_status,
    )
    if initial_accumulator_value <= 0:
      raise ValueError(
          f"Argument `initial_accumulator_value` must be a positive float. "
          f"Received: {initial_accumulator_value}")
    self.initial_accumulator_value = initial_accumulator_value
  def _slot_names(self) -> List[Text]:
    return ["accumulators"]
  def _slot_initializers(self) -> List[init_ops_v2.Initializer]:
    return [init_ops_v2.Constant(self.initial_accumulator_value)]
  def _set_optimization_parameters(
      self, parameters: optimization_parameters_pb2.OptimizationParameters):
    super()._set_optimization_parameters(parameters)
    parameters.adagrad.SetInParent()
  def _load(self) -> Callable[..., ops.Operation]:
    return tpu_ops.load_tpu_embedding_adagrad_parameters
  def _retrieve(self) -> Callable[..., core.Tensor]:
    return tpu_ops.retrieve_tpu_embedding_adagrad_parameters
