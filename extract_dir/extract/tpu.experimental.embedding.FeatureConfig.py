@tf_export("tpu.experimental.embedding.FeatureConfig")
class FeatureConfig:
  """Configuration data for one embedding feature.
  This class holds the configuration data for a single embedding feature. The
  main use is to assign features to `tf.tpu.experimental.embedding.TableConfig`s
  via the table parameter:
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
  You can also specify the output shape for each feature. The output shape
  should be the expected activation shape excluding the table dimension. For
  dense and sparse tensor, the output shape should be the same as the input
  shape excluding the last dimension. For ragged tensor, the output shape can
  mismatch the input shape.
  NOTE: The `max_sequence_length` will be only used when the input tensor has
  rank 2 and the `output_shape` is not set in the feature config.
  When feeding features into `embedding.enqueue` they can be `tf.Tensor`s,
  `tf.SparseTensor`s or `tf.RaggedTensor`s. When the argument
  `max_sequence_length` is 0, the default, you should expect a output of
  `embedding.dequeue` for this feature of shape `(batch_size, dim)`. If
  `max_sequence_length` is greater than 0, the feature is embedded as a sequence
  and padded up to the given length. The shape of the output for this feature
  will be `(batch_size, max_sequence_length, dim)`.
  """
  def __init__(self,
               table: TableConfig,
               max_sequence_length: int = 0,
               validate_weights_and_indices: bool = True,
               output_shape: Optional[Union[List[int], TensorShape]] = None,
               name: Optional[Text] = None):
    """Feature configuration.
    Args:
      table: An instance of `tf.tpu.experimental.embedding.TableConfig`,
        describing the table in which this feature should be looked up.
      max_sequence_length: If positive, the feature is a sequence feature with
        the corresponding maximum sequence length. If the sequence is longer
        than this, it will be truncated. If 0, the feature is not a sequence
        feature.
      validate_weights_and_indices: If true, uses safe_embedding_lookup during
        serving which ensures there are no empty rows and all weights and ids
        are positive at the expense of extra compute cost.
      output_shape: Optional argument to config the output shape of the feature
        activation. If provided, the feature feeding to the `embedding.enqueue`
        has to match the shape (for ragged tensor, the input shape and output
        shape can mismatch). If not provided, the shape can be either provided
        to the `embedding.build` or auto detected at the runtime.
      name: An optional name for the feature, useful for debugging.
    Returns:
      `FeatureConfig`.
    Raises:
      ValueError: if `table` is not an instance of
        `tf.tpu.experimental.embedding.TableConfig`.
      ValueError: if `max_sequence_length` not an integer or is negative.
    """
    if not isinstance(table, TableConfig):
      raise ValueError(f"Argument `table` has invalid type {type(table)}. "
                       "Expected `tf.tpu.experimental.embedding.TableConfig`.")
    if not isinstance(max_sequence_length, int) or max_sequence_length < 0:
      raise ValueError(
          f"Argument `max_sequence_length` must be an int and must be >= 0. "
          f"Received: {max_sequence_length}")
    self.table = table
    self.max_sequence_length = max_sequence_length
    self.name = name
    self.output_shape = TensorShape(output_shape)
    if not isinstance(
        validate_weights_and_indices, bool):
      raise ValueError(
          f"Argument `validate_weights_and_indices` must be a boolean. "
          f"Received: {validate_weights_and_indices}")
    self.validate_weights_and_indices = validate_weights_and_indices
  def __repr__(self):
    return ("FeatureConfig(table={table!r}, "
            "max_sequence_length={max_sequence_length!r}, "
            "validate_weights_and_indices={validate_weights_and_indices!r}, "
            "output_shape={output_shape!r}, name={name!r})".format(
                table=self.table,
                max_sequence_length=self.max_sequence_length,
                validate_weights_and_indices=self.validate_weights_and_indices,
                output_shape=self.output_shape,
                name=self.name))
