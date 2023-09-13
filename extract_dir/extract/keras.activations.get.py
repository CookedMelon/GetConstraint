@keras_export("keras.activations.get")
@tf.__internal__.dispatch.add_dispatch_support
def get(identifier):
    """Returns function.
    Args:
        identifier: Function or string
    Returns:
        Function corresponding to the input string or input function.
    Example:
    >>> tf.keras.activations.get('softmax')
     <function softmax at 0x1222a3d90>
    >>> tf.keras.activations.get(tf.keras.activations.softmax)
     <function softmax at 0x1222a3d90>
    >>> tf.keras.activations.get(None)
     <function linear at 0x1239596a8>
    >>> tf.keras.activations.get(abs)
     <built-in function abs>
    >>> tf.keras.activations.get('abcd')
    Traceback (most recent call last):
    ...
    ValueError: Unknown activation function:abcd
    Raises:
        ValueError: Input is an unknown function or string, i.e., the input does
        not denote any defined function.
    """
    if identifier is None:
        return linear
    if isinstance(identifier, (str, dict)):
        use_legacy_format = (
            "module" not in identifier
            if isinstance(identifier, dict)
            else False
        )
        return deserialize(identifier, use_legacy_format=use_legacy_format)
    elif callable(identifier):
        return identifier
    raise TypeError(
        f"Could not interpret activation function identifier: {identifier}"
    )
