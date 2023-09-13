@keras_export("keras.constraints.get")
def get(identifier):
    """Retrieves a Keras constraint function."""
    if identifier is None:
        return None
    if isinstance(identifier, dict):
        use_legacy_format = "module" not in identifier
        return deserialize(identifier, use_legacy_format=use_legacy_format)
    elif isinstance(identifier, str):
        config = {"class_name": str(identifier), "config": {}}
        return get(config)
    elif callable(identifier):
        return identifier
    else:
        raise ValueError(
            f"Could not interpret constraint function identifier: {identifier}"
        )
