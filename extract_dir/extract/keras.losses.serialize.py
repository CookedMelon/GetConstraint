@keras_export("keras.losses.serialize")
def serialize(loss, use_legacy_format=False):
    """Serializes loss function or `Loss` instance.
    Args:
      loss: A Keras `Loss` instance or a loss function.
    Returns:
      Loss configuration dictionary.
    """
    if use_legacy_format:
        return legacy_serialization.serialize_keras_object(loss)
    return serialize_keras_object(loss)
