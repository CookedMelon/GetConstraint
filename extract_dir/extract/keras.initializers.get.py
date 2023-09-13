@keras_export("keras.initializers.get")
def get(identifier):
    """Retrieve a Keras initializer by the identifier.
    The `identifier` may be the string name of a initializers function or class
    (case-sensitively).
    >>> identifier = 'Ones'
    >>> tf.keras.initializers.deserialize(identifier)
    <...keras.initializers.initializers.Ones...>
    You can also specify `config` of the initializer to this function by passing
    dict containing `class_name` and `config` as an identifier. Also note that
    the `class_name` must map to a `Initializer` class.
    >>> cfg = {'class_name': 'Ones', 'config': {}}
    >>> tf.keras.initializers.deserialize(cfg)
    <...keras.initializers.initializers.Ones...>
    In the case that the `identifier` is a class, this method will return a new
    instance of the class by its constructor.
    Args:
      identifier: String or dict that contains the initializer name or
        configurations.
    Returns:
      Initializer instance base on the input identifier.
    Raises:
      ValueError: If the input identifier is not a supported type or in a bad
        format.
    """
    if identifier is None:
        return None
    if isinstance(identifier, dict):
        use_legacy_format = "module" not in identifier
        return deserialize(identifier, use_legacy_format=use_legacy_format)
    elif isinstance(identifier, str):
        config = {"class_name": str(identifier), "config": {}}
        return get(config)
    elif callable(identifier):
        if inspect.isclass(identifier):
            identifier = identifier()
        return identifier
    else:
        raise ValueError(
            "Could not interpret initializer identifier: " + str(identifier)
        )
