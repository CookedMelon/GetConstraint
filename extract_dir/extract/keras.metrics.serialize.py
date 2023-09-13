@keras_export("keras.metrics.serialize")
def serialize(metric, use_legacy_format=False):
    """Serializes metric function or `Metric` instance.
    Args:
      metric: A Keras `Metric` instance or a metric function.
    Returns:
      Metric configuration dictionary.
    """
    if use_legacy_format:
        return legacy_serialization.serialize_keras_object(metric)
    return serialize_keras_object(metric)
