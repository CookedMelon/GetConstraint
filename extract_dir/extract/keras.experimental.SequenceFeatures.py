@keras_export("keras.experimental.SequenceFeatures")
class SequenceFeatures(kfc._BaseFeaturesLayer):
    """A layer for sequence input.
    All `feature_columns` must be sequence dense columns with the same
    `sequence_length`. The output of this method can be fed into sequence
    networks, such as RNN.
    The output of this method is a 3D `Tensor` of shape `[batch_size, T, D]`.
    `T` is the maximum sequence length for this batch, which could differ from
    batch to batch.
    If multiple `feature_columns` are given with `Di` `num_elements` each, their
    outputs are concatenated. So, the final `Tensor` has shape
    `[batch_size, T, D0 + D1 + ... + Dn]`.
    Example:
    ```python
    import tensorflow as tf
    # Behavior of some cells or feature columns may depend on whether we are in
    # training or inference mode, e.g. applying dropout.
    training = True
    rating = tf.feature_column.sequence_numeric_column('rating')
    watches = tf.feature_column.sequence_categorical_column_with_identity(
        'watches', num_buckets=1000)
    watches_embedding = tf.feature_column.embedding_column(watches,
                                                dimension=10)
    columns = [rating, watches_embedding]
    features = {
     'rating': tf.sparse.from_dense([[1.0,1.1, 0, 0, 0],
                                                 [2.0,2.1,2.2, 2.3, 2.5]]),
     'watches': tf.sparse.from_dense([[2, 85, 0, 0, 0],[33,78, 2, 73, 1]])
    }
    sequence_input_layer = tf.keras.experimental.SequenceFeatures(columns)
    sequence_input, sequence_length = sequence_input_layer(
       features, training=training)
    sequence_length_mask = tf.sequence_mask(sequence_length)
    hidden_size = 32
    rnn_cell = tf.keras.layers.SimpleRNNCell(hidden_size)
    rnn_layer = tf.keras.layers.RNN(rnn_cell)
    outputs, state = rnn_layer(sequence_input, mask=sequence_length_mask)
    ```
    """
    def __init__(self, feature_columns, trainable=True, name=None, **kwargs):
        """ "Constructs a SequenceFeatures layer.
        Args:
          feature_columns: An iterable of dense sequence columns. Valid columns
            are
            - `embedding_column` that wraps a
              `sequence_categorical_column_with_*`
            - `sequence_numeric_column`.
          trainable: Boolean, whether the layer's variables will be updated via
            gradient descent during training.
          name: Name to give to the SequenceFeatures.
          **kwargs: Keyword arguments to construct a layer.
        Raises:
          ValueError: If any of the `feature_columns` is not a
            `SequenceDenseColumn`.
        """
        super().__init__(
            feature_columns=feature_columns,
            trainable=trainable,
            name=name,
            expected_column_type=tf.__internal__.feature_column.SequenceDenseColumn,  # noqa: E501
            **kwargs
        )
    @property
    def _is_feature_layer(self):
        return True
    def _target_shape(self, input_shape, total_elements):
        return (input_shape[0], input_shape[1], total_elements)
    def call(self, features, training=None):
        """Returns sequence input corresponding to the `feature_columns`.
        Args:
          features: A dict mapping keys to tensors.
          training: Python boolean or None, indicating whether to the layer is
            being run in training mode. This argument is passed to the call
            method of any `FeatureColumn` that takes a `training` argument. For
            example, if a `FeatureColumn` performed dropout, the column could
            expose a `training` argument to control whether the dropout should
            be applied. If `None`, defaults to
            `tf.keras.backend.learning_phase()`.
        Returns:
          An `(input_layer, sequence_length)` tuple where:
          - input_layer: A float `Tensor` of shape `[batch_size, T, D]`.
              `T` is the maximum sequence length for this batch, which could
              differ from batch to batch. `D` is the sum of `num_elements` for
              all `feature_columns`.
          - sequence_length: An int `Tensor` of shape `[batch_size]`. The
            sequence length for each example.
        Raises:
          ValueError: If features are not a dictionary.
        """
        if not isinstance(features, dict):
            raise ValueError(
                "We expected a dictionary here. Instead we got: ", features
            )
        if training is None:
            training = backend.learning_phase()
        transformation_cache = (
            tf.__internal__.feature_column.FeatureTransformationCache(features)
        )
        output_tensors = []
        sequence_lengths = []
        for column in self._feature_columns:
            with backend.name_scope(column.name):
                try:
                    (
                        dense_tensor,
                        sequence_length,
                    ) = column.get_sequence_dense_tensor(
                        transformation_cache,
                        self._state_manager,
                        training=training,
                    )
                except TypeError:
                    (
                        dense_tensor,
                        sequence_length,
                    ) = column.get_sequence_dense_tensor(
                        transformation_cache, self._state_manager
                    )
                # Flattens the final dimension to produce a 3D Tensor.
                output_tensors.append(
                    self._process_dense_tensor(column, dense_tensor)
                )
                sequence_lengths.append(sequence_length)
        # Check and process sequence lengths.
        kfc._verify_static_batch_size_equality(
            sequence_lengths, self._feature_columns
        )
        sequence_length = _assert_all_equal_and_return(sequence_lengths)
        return self._verify_and_concat_tensors(output_tensors), sequence_length
