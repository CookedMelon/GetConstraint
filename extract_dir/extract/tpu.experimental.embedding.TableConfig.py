@tf_export("tpu.experimental.embedding.TableConfig")
class TableConfig:
  """Configuration data for one embedding table.
  This class holds the configuration data for a single embedding table. It is
  used as the `table` parameter of a
  `tf.tpu.experimental.embedding.FeatureConfig`. Multiple
  `tf.tpu.experimental.embedding.FeatureConfig` objects can use the same
  `tf.tpu.experimental.embedding.TableConfig` object. In this case a shared
  table will be created for those feature lookups.
  ```python
  table_config_one = tf.tpu.experimental.embedding.TableConfig(
      vocabulary_size=...,
      dim=...)
  table_config_two = tf.tpu.experimental.embedding.TableConfig(
      vocabulary_size=...,
      dim=...)
  feature_config = {
      'feature_one': tf.tpu.experimental.embedding.FeatureConfig(
          table=table_config_one),
      'feature_two': tf.tpu.experimental.embedding.FeatureConfig(
          table=table_config_one),
      'feature_three': tf.tpu.experimental.embedding.FeatureConfig(
          table=table_config_two)}
  embedding = tf.tpu.experimental.embedding.TPUEmbedding(
      feature_config=feature_config,
      batch_size=...
      optimizer=tf.tpu.experimental.embedding.Adam(0.1))
  ```
  The above configuration has 2 tables, and three features. The first two
  features will be looked up in the first table and the third feature will be
  looked up in the second table.
  """
  def __init__(self,
               vocabulary_size: int,
               dim: int,
               initializer: Optional[Callable[[Any], None]] = None,
               optimizer: Optional[_Optimizer] = None,
               combiner: Text = "mean",
               name: Optional[Text] = None,
               quantization_config: QuantizationConfig = None):
    """Embedding table configuration.
    Args:
      vocabulary_size: Size of the table's vocabulary (number of rows).
      dim: The embedding dimension (width) of the table.
      initializer: A callable initializer taking one parameter, the shape of the
        variable that will be initialized. Will be called once per task, to
        initialize that task's shard of the embedding table. If not specified,
        defaults to `truncated_normal_initializer` with mean `0.0` and standard
        deviation `1/sqrt(dim)`.
      optimizer: An optional instance of an optimizer parameters class, instance
        of one of `tf.tpu.experimental.embedding.SGD`,
        `tf.tpu.experimental.embedding.Adagrad` or
        `tf.tpu.experimental.embedding.Adam`. If set will override the global
        optimizer passed to `tf.tpu.experimental.embedding.TPUEmbedding`.
      combiner: A string specifying how to reduce if there are multiple entries
        in a single row. Currently 'mean', 'sqrtn', 'sum' are supported, with
        'mean' the default. 'sqrtn' often achieves good accuracy, in particular
        with bag-of-words columns. For more information, see
        `tf.nn.embedding_lookup_sparse`.
      name: An optional string used to name the table. Useful for debugging.
      quantization_config: The simulated quantization config. An instance of
        `tf.tpu.experimental.embedding.QuantizationConfig`. See the class for
        more documentation.
    Returns:
      `TableConfig`.
    Raises:
      ValueError: if `vocabulary_size` is not a positive integer.
      ValueError: if `dim` is not a positive integer.
      ValueError: if `initializer` is specified and is not callable.
      ValueError: if `combiner` is not supported.
    """
    if not isinstance(vocabulary_size, int) or vocabulary_size < 1:
      raise ValueError(
          f"Argument `vocabulary_size` must be an int and must be >= 1. "
          f"Received: {vocabulary_size}")
    if not isinstance(dim, int) or dim < 1:
      raise ValueError(
          f"Argument `dim` (embedding dimension) "
          f"must be an int and must be >= 1. Received: {dim}")
    if (initializer is not None) and (not callable(initializer)):
      raise ValueError(
          f"Argument `initializer` must be a callable (or None). "
          f"Received: {initializer}")
    if initializer is None:
      initializer = init_ops_v2.TruncatedNormal(mean=0.0,
                                                stddev=1/math.sqrt(dim))
    accepted_combiners = ("mean", "sum", "sqrtn")
    if combiner not in accepted_combiners:
      raise ValueError(
          f"Argument `combiner` must be one of {accepted_combiners}. "
          f"Received: {combiner}")
    self.vocabulary_size = vocabulary_size
    self.dim = dim
    self.initializer = initializer
    self.optimizer = optimizer
    self.combiner = combiner
    self.name = name
    self.quantization_config = quantization_config
  def __repr__(self):
    # If using the default initializer, just print "None" for clarity.
    initializer = self.initializer
    if isinstance(initializer, init_ops_v2.TruncatedNormal):
      # PY2 type checking can't infer type of initializer even after if.
      initializer = typing.cast(init_ops_v2.TruncatedNormal, initializer)
      if (initializer.mean == 0.0
          and math.isclose(initializer.stddev, 1/math.sqrt(self.dim))):
        initializer = None
    return ("TableConfig(vocabulary_size={vocabulary_size!r}, dim={dim!r}, "
            "initializer={initializer!r}, optimizer={optimizer!r}, "
            "combiner={combiner!r}, name={name!r}, "
            "quantization_config={quantization!r})".format(
                vocabulary_size=self.vocabulary_size,
                dim=self.dim,
                initializer=initializer,
                optimizer=self.optimizer,
                combiner=self.combiner,
                name=self.name,
                quantization=self.quantization_config,
            ))
  def _set_table_descriptor(
      self,
      table_descriptor: tpu_embedding_configuration_pb2
      .TPUEmbeddingConfiguration.TableDescriptor,
      num_hosts: int,
      learning_rate_index: Dict[Callable[[], Any], int]):
    """Set the table descriptor from the table data."""
    table_descriptor.name = self.name
    # For small tables, we pad to the number of hosts so that at least one
    # id will be assigned to each host.
    table_descriptor.vocabulary_size = max(self.vocabulary_size, num_hosts)
    table_descriptor.dimension = self.dim
    parameters = table_descriptor.optimization_parameters
    # We handle the learning rate separately here and don't allow the
    # optimization class to handle this, as it doesn't know about dynamic
    # rates.
    if callable(self.optimizer.learning_rate):
      parameters.learning_rate.dynamic.tag = (
          learning_rate_index[self.optimizer.learning_rate])
    else:
      parameters.learning_rate.constant = self.optimizer.learning_rate
    if self.optimizer.low_dimensional_packing_status:
      parameters.low_dimensional_packing_status = (
          optimization_parameters_pb2.LowDimensionalPackingStatus.Status.ENABLED
      )
    # Use optimizer to handle the rest of the parameters.
    self.optimizer._set_optimization_parameters(parameters)  # pylint: disable=protected-access
    if self.quantization_config:
      self.quantization_config._set_optimization_parameters(parameters)  # pylint: disable=protected-access
