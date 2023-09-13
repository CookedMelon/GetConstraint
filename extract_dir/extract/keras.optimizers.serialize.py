@keras_export("keras.optimizers.serialize")
def serialize(optimizer, use_legacy_format=False):
    """Serialize the optimizer configuration to JSON compatible python dict.
    The configuration can be used for persistence and reconstruct the
    `Optimizer` instance again.
    >>> tf.keras.optimizers.serialize(tf.keras.optimizers.legacy.SGD())
    {'module': 'keras.optimizers.legacy', 'class_name': 'SGD', 'config': {'name': 'SGD', 'learning_rate': 0.01, 'decay': 0.0, 'momentum': 0.0, 'nesterov': False}, 'registered_name': None}"""  # noqa: E501
    """
    Args:
      optimizer: An `Optimizer` instance to serialize.
    Returns:
      Python dict which contains the configuration of the input optimizer.
    """
    if use_legacy_format:
        return legacy_serialization.serialize_keras_object(optimizer)
    return serialize_keras_object(optimizer)
