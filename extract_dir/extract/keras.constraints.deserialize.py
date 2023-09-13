@keras_export("keras.constraints.deserialize")
def deserialize(config, custom_objects=None, use_legacy_format=False):
    if use_legacy_format:
        return legacy_serialization.deserialize_keras_object(
            config,
            module_objects=globals(),
            custom_objects=custom_objects,
            printable_module_name="constraint",
        )
    return deserialize_keras_object(
        config,
        module_objects=globals(),
        custom_objects=custom_objects,
        printable_module_name="constraint",
    )
