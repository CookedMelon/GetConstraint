@keras_export("keras.backend.floatx")
def floatx():
    """Returns the default float type, as a string.
    E.g. `'float16'`, `'float32'`, `'float64'`.
    Returns:
        String, the current default float type.
    Example:
    >>> tf.keras.backend.floatx()
    'float32'
    """
    return _FLOATX
