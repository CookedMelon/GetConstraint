@keras_export("keras.initializers.deserialize")
def deserialize(config, custom_objects=None, use_legacy_format=False):
    """Return an `Initializer` object from its config."""
    populate_deserializable_objects()
    if use_legacy_format:
        return legacy_serialization.deserialize_keras_object(
            config,
            module_objects=LOCAL.ALL_OBJECTS,
            custom_objects=custom_objects,
            printable_module_name="initializer",
        )
    return serialization_lib.deserialize_keras_object(
        config,
        module_objects=LOCAL.ALL_OBJECTS,
        custom_objects=custom_objects,
        printable_module_name="initializer",
    )
