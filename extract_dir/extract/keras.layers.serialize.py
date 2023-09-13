@keras_export("keras.layers.serialize")
def serialize(layer, use_legacy_format=False):
    """Serializes a `Layer` object into a JSON-compatible representation.
    Args:
      layer: The `Layer` object to serialize.
    Returns:
      A JSON-serializable dict representing the object's config.
    Example:
    ```python
    from pprint import pprint
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(16,)))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    pprint(tf.keras.layers.serialize(model))
    # prints the configuration of the model, as a dict.
    """
    if isinstance(layer, base_metric.Metric):
        raise ValueError(
            f"Cannot serialize {layer} since it is a metric. "
            "Please use the `keras.metrics.serialize()` and "
            "`keras.metrics.deserialize()` APIs to serialize "
            "and deserialize metrics."
        )
    if use_legacy_format:
        return legacy_serialization.serialize_keras_object(layer)
    return serialization_lib.serialize_keras_object(layer)
