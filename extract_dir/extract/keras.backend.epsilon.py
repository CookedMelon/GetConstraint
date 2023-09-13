@keras_export("keras.backend.epsilon")
@tf.__internal__.dispatch.add_dispatch_support
def epsilon():
    """Returns the value of the fuzz factor used in numeric expressions.
    Returns:
        A float.
    Example:
    >>> tf.keras.backend.epsilon()
    1e-07
    """
    return _EPSILON
