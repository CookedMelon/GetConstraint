@keras_export("keras.utils.legacy.serialize_keras_object")
def serialize_keras_object(instance):
    """Serialize a Keras object into a JSON-compatible representation.
    Calls to `serialize_keras_object` while underneath the
    `SharedObjectSavingScope` context manager will cause any objects re-used
    across multiple layers to be saved with a special shared object ID. This
    allows the network to be re-created properly during deserialization.
    Args:
      instance: The object to serialize.
    Returns:
      A dict-like, JSON-compatible representation of the object's config.
    """
    from keras.saving import object_registration
    _, instance = tf.__internal__.decorator.unwrap(instance)
    if instance is None:
        return None
    if hasattr(instance, "get_config"):
        name = object_registration.get_registered_name(instance.__class__)
        try:
            config = instance.get_config()
        except NotImplementedError as e:
            if _SKIP_FAILED_SERIALIZATION:
                return serialize_keras_class_and_config(
                    name, {_LAYER_UNDEFINED_CONFIG_KEY: True}
                )
            raise e
        serialization_config = {}
        for key, item in config.items():
            if isinstance(item, str):
                serialization_config[key] = item
                continue
            # Any object of a different type needs to be converted to string or
            # dict for serialization (e.g. custom functions, custom classes)
            try:
                serialized_item = serialize_keras_object(item)
                if isinstance(serialized_item, dict) and not isinstance(
                    item, dict
                ):
                    serialized_item["__passive_serialization__"] = True
                serialization_config[key] = serialized_item
            except ValueError:
                serialization_config[key] = item
        name = object_registration.get_registered_name(instance.__class__)
        return serialize_keras_class_and_config(
            name, serialization_config, instance
        )
    if hasattr(instance, "__name__"):
        return object_registration.get_registered_name(instance)
    raise ValueError(
        f"Cannot serialize {instance} because it doesn't implement "
        "`get_config()`."
    )
