@keras_export("keras.layers.DenseFeatures", v1=[])
class DenseFeatures(dense_features.DenseFeatures):
    """A layer that produces a dense `Tensor` based on given `feature_columns`.
    Generally a single example in training data is described with
    FeatureColumns.  At the first layer of the model, this column oriented data
    should be converted to a single `Tensor`.
    This layer can be called multiple times with different features.
    This is the V2 version of this layer that uses name_scopes to create
    variables instead of variable_scopes. But this approach currently lacks
    support for partitioned variables. In that case, use the V1 version instead.
    Example:
    ```python
    price = tf.feature_column.numeric_column('price')
    keywords_embedded = tf.feature_column.embedding_column(
        tf.feature_column.categorical_column_with_hash_bucket("keywords",
                                                              10000),
        dimensions=16)
    columns = [price, keywords_embedded, ...]
    feature_layer = tf.keras.layers.DenseFeatures(columns)
    features = tf.io.parse_example(
        ..., features=tf.feature_column.make_parse_example_spec(columns))
    dense_tensor = feature_layer(features)
    for units in [128, 64, 32]:
      dense_tensor = tf.keras.layers.Dense(units, activation='relu')(
        dense_tensor)
    prediction = tf.keras.layers.Dense(1)(dense_tensor)
    ```
    """
    def __init__(self, feature_columns, trainable=True, name=None, **kwargs):
        """Creates a DenseFeatures object.
        Args:
          feature_columns: An iterable containing the FeatureColumns to use as
            inputs to your model. All items should be instances of classes
            derived from `DenseColumn` such as `numeric_column`,
            `embedding_column`, `bucketized_column`, `indicator_column`. If you
            have categorical features, you can wrap them with an
            `embedding_column` or `indicator_column`.
          trainable:  Boolean, whether the layer's variables will be updated via
            gradient descent during training.
          name: Name to give to the DenseFeatures.
          **kwargs: Keyword arguments to construct a layer.
        Raises:
          ValueError: if an item in `feature_columns` is not a `DenseColumn`.
        """
        super().__init__(
            feature_columns=feature_columns,
            trainable=trainable,
            name=name,
            **kwargs
        )
        self._state_manager = _StateManagerImplV2(self, self.trainable)
    def build(self, _):
        for column in self._feature_columns:
            with tf.name_scope(column.name):
                column.create_state(self._state_manager)
        # We would like to call Layer.build and not _DenseFeaturesHelper.build.
        super(kfc._BaseFeaturesLayer, self).build(None)
