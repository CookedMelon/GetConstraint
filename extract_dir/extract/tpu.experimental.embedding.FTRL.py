@tf_export("tpu.experimental.embedding.FTRL")
class FTRL(_Optimizer):
  """Optimization parameters for FTRL with TPU embeddings.
  See Algorithm 1 of this
  [paper](https://research.google.com/pubs/archive/41159.pdf).
  Pass this to `tf.tpu.experimental.embedding.TPUEmbedding` via the `optimizer`
  argument to set the global optimizer and its parameters:
  ```python
  embedding = tf.tpu.experimental.embedding.TPUEmbedding(
      ...
      optimizer=tf.tpu.experimental.embedding.FTRL(0.1))
  ```
  This can also be used in a `tf.tpu.experimental.embedding.TableConfig` as the
  optimizer parameter to set a table specific optimizer. This will override the
  optimizer and parameters for global embedding optimizer defined above:
  ```python
  table_one = tf.tpu.experimental.embedding.TableConfig(
      vocabulary_size=...,
      dim=...,
      optimizer=tf.tpu.experimental.embedding.FTRL(0.2))
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
      optimizer=tf.tpu.experimental.embedding.FTRL(0.1))
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
      learning_rate_power: float = -0.5,
      l1_regularization_strength: float = 0.0,
      l2_regularization_strength: float = 0.0,
      beta: float = 0.0,
      initial_accumulator_value: float = 0.1,
      use_gradient_accumulation: bool = True,
      clip_weight_min: Optional[float] = None,
      clip_weight_max: Optional[float] = None,
      weight_decay_factor: Optional[float] = None,
      multiply_weight_decay_factor_by_learning_rate: bool = None,
      slot_variable_creation_fn: Optional[SlotVarCreationFnType] = None,
      clipvalue: Optional[ClipValueType] = None,
      multiply_linear_by_learning_rate: bool = False,
      allow_zero_accumulator: bool = False,
      low_dimensional_packing_status: bool = False,
  ):
    """Optimization parameters for Adagrad.
    Args:
      learning_rate: The learning rate. It should be a floating point value or a
        callable taking no arguments for a dynamic learning rate.
      learning_rate_power: A float value, must be less or equal to zero.
        Controls how the learning rate decreases during training. Use zero for a
        fixed learning rate.
      l1_regularization_strength: A float value, must be greater than or equal
        to zero.
      l2_regularization_strength: A float value, must be greater than or equal
        to zero.
      beta: A float value, representing the beta value from the paper.
      initial_accumulator_value: The starting value for accumulators. Only zero
        or positive values are allowed.
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
      multiply_linear_by_learning_rate: If set to True, a modified formula is
        used for FTRL that treats the "linear" accumulator as being
        pre-multiplied by the learning rate (i.e., the accumulator named
        "linear" actually stores "linear * learning_rate"). Other than
        checkpoint compatibility, this is mathematically equivalent for a static
        learning rate; for a dynamic learning rate, it is nearly the same as
        long as the learning rate does not change quickly. The benefit of this
        is that the modified formula handles zero and near-zero learning rates
        without producing NaNs, improving flexibility for learning rate ramp-up.
      allow_zero_accumulator: If set to True, changes some internal formulas to
        allow zero and near-zero accumulator values at the cost of some
        performance; this only needs to be set if you are using an initial
        accumulator value of zero, which is uncommon.
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
    self.learning_rate_power = learning_rate_power
    self.l1_regularization_strength = l1_regularization_strength
    self.l2_regularization_strength = l2_regularization_strength
    self.beta = beta
    self.multiply_linear_by_learning_rate = multiply_linear_by_learning_rate
    self.allow_zero_accumulator = allow_zero_accumulator
  def _slot_names(self) -> List[Text]:
    return ["accumulators", "linears"]
  def _slot_initializers(self) -> List[init_ops_v2.Initializer]:
    return [
        init_ops_v2.Constant(self.initial_accumulator_value),
        init_ops_v2.Constant()
    ]
  def _set_optimization_parameters(
      self, parameters: optimization_parameters_pb2.OptimizationParameters):
    super()._set_optimization_parameters(parameters)
    ftrl = parameters.ftrl
    ftrl.l1 = self.l1_regularization_strength
    ftrl.l2 = self.l2_regularization_strength
    ftrl.lr_power = self.learning_rate_power
    ftrl.beta = self.beta
    ftrl.multiply_linear_by_lr = self.multiply_linear_by_learning_rate
    ftrl.allow_zero_accumulator = self.allow_zero_accumulator
  def _load(self) -> Callable[..., ops.Operation]:
    return tpu_ops.load_tpu_embedding_ftrl_parameters
  def _retrieve(self) -> Callable[..., core.Tensor]:
    return tpu_ops.retrieve_tpu_embedding_ftrl_parameters
