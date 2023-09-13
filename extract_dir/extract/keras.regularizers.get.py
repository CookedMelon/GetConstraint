@keras_export("keras.regularizers.get")
def get(identifier):
    """Retrieve a regularizer instance from a config or identifier."""
    if identifier is None:
        return None
    if isinstance(identifier, dict):
        use_legacy_format = "module" not in identifier
        return deserialize(identifier, use_legacy_format=use_legacy_format)
    elif isinstance(identifier, str):
        return deserialize(str(identifier))
    elif callable(identifier):
        return identifier
    else:
        raise ValueError(
            f"Could not interpret regularizer identifier: {identifier}"
        )
