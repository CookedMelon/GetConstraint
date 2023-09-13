@keras_export("keras.metrics.deserialize")
def deserialize(config, custom_objects=None, use_legacy_format=False):
    """Deserializes a serialized metric class/function instance.
    Args:
      config: Metric configuration.
      custom_objects: Optional dictionary mapping names (strings) to custom
        objects (classes and functions) to be considered during deserialization.
    Returns:
        A Keras `Metric` instance or a metric function.
    """
    if use_legacy_format:
        return legacy_serialization.deserialize_keras_object(
            config,
            module_objects=globals(),
            custom_objects=custom_objects,
            printable_module_name="metric function",
        )
    return deserialize_keras_object(
        config,
        module_objects=globals(),
        custom_objects=custom_objects,
        printable_module_name="metric function",
    )
