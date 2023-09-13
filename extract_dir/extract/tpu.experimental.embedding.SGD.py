@tf_export("tpu.experimental.embedding.SGD")
class SGD(_Optimizer):
  """Optimization parameters for stochastic gradient descent for TPU embeddings.
  Pass this to `tf.tpu.experimental.embedding.TPUEmbedding` via the `optimizer`
  argument to set the global optimizer and its parameters:
  ```
  embedding = tf.tpu.experimental.embedding.TPUEmbedding(
      ...
      optimizer=tf.tpu.experimental.embedding.SGD(0.1))
  ```
  This can also be used in a `tf.tpu.experimental.embedding.TableConfig` as the
  optimizer parameter to set a table specific optimizer. This will override the
  optimizer and parameters for global embedding optimizer defined above:
  ```
  table_one = tf.tpu.experimental.embedding.TableConfig(
      vocabulary_size=...,
      dim=...,
      optimizer=tf.tpu.experimental.embedding.SGD(0.2))
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
      optimizer=tf.tpu.experimental.embedding.SGD(0.1))
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
      learning_rate: Union[float, Callable[[], float]] = 0.01,
      use_gradient_accumulation: bool = True,
      clip_weight_min: Optional[float] = None,
      clip_weight_max: Optional[float] = None,
      weight_decay_factor: Optional[float] = None,
      multiply_weight_decay_factor_by_learning_rate: bool = None,
      clipvalue: Optional[ClipValueType] = None,
      low_dimensional_packing_status: bool = False,
  ):
    """Optimization parameters for stochastic gradient descent.
    Args:
      learning_rate: The learning rate. It should be a floating point value or a
        callable taking no arguments for a dynamic learning rate.
      use_gradient_accumulation: setting this to `False` makes embedding
        gradients calculation less accurate but faster.
      clip_weight_min: the minimum value to clip by; None means -infinity.
      clip_weight_max: the maximum value to clip by; None means +infinity.
      weight_decay_factor: amount of weight decay to apply; None means that the
        weights are not decayed. Weights are decayed by multiplying the weight
        by this factor each step.
      multiply_weight_decay_factor_by_learning_rate: if true,
        `weight_decay_factor` is multiplied by the current learning rate.
      clipvalue: Controls clipping of the gradient. Set to either a single
        positive scalar value to get clipping or a tiple of scalar values (min,
        max) to set a separate maximum or minimum. If one of the two entries is
        None, then there will be no clipping that direction. Note if this is
        set, you may see a decrease in performance as  gradient accumulation
        will be enabled (it is normally off for SGD as it has no affect on
        accuracy). See
        'tensorflow/core/protobuf/tpu/optimization_parameters.proto' for more
        information on gradient accumulation and its impact on tpu embeddings.
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
        None,
        low_dimensional_packing_status,
    )
  def _slot_names(self) -> List[Text]:
    return []
  def _slot_initializers(self) -> List[init_ops_v2.Initializer]:
    return []
  def _set_optimization_parameters(
      self, parameters: optimization_parameters_pb2.OptimizationParameters):
    super()._set_optimization_parameters(parameters)
    parameters.stochastic_gradient_descent.SetInParent()
  def _load(self) -> Callable[..., ops.Operation]:
    return tpu_ops.load_tpu_embedding_stochastic_gradient_descent_parameters
  def _retrieve(self) -> Callable[..., core.Tensor]:
    return tpu_ops.retrieve_tpu_embedding_stochastic_gradient_descent_parameters
