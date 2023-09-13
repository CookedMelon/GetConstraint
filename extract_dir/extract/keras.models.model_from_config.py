@keras_export("keras.models.model_from_config")
def model_from_config(config, custom_objects=None):
    """Instantiates a Keras model from its config.
    Usage:
    ```
    # for a Functional API model
    tf.keras.Model().from_config(model.get_config())
    # for a Sequential model
    tf.keras.Sequential().from_config(model.get_config())
    ```
    Args:
        config: Configuration dictionary.
        custom_objects: Optional dictionary mapping names
            (strings) to custom classes or functions to be
            considered during deserialization.
    Returns:
        A Keras model instance (uncompiled).
    Raises:
        TypeError: if `config` is not a dictionary.
    """
    if isinstance(config, list):
        raise TypeError(
            "`model_from_config` expects a dictionary, not a list. "
            f"Received: config={config}. Did you meant to use "
            "`Sequential.from_config(config)`?"
        )
    from keras import layers
    global MODULE_OBJECTS
    if not hasattr(MODULE_OBJECTS, "ALL_OBJECTS"):
        layers.serialization.populate_deserializable_objects()
        MODULE_OBJECTS.ALL_OBJECTS = layers.serialization.LOCAL.ALL_OBJECTS
    return serialization.deserialize_keras_object(
        config,
        module_objects=MODULE_OBJECTS.ALL_OBJECTS,
        custom_objects=custom_objects,
        printable_module_name="layer",
    )
