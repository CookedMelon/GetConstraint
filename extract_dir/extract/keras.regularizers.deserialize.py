@keras_export("keras.regularizers.deserialize")
def deserialize(config, custom_objects=None, use_legacy_format=False):
    if config == "l1_l2":
        # Special case necessary since the defaults used for "l1_l2" (string)
        # differ from those of the L1L2 class.
        return L1L2(l1=0.01, l2=0.01)
    if use_legacy_format:
        return legacy_serialization.deserialize_keras_object(
            config,
            module_objects=globals(),
            custom_objects=custom_objects,
            printable_module_name="regularizer",
        )
    return deserialize_keras_object(
        config,
        module_objects=globals(),
        custom_objects=custom_objects,
        printable_module_name="regularizer",
    )
