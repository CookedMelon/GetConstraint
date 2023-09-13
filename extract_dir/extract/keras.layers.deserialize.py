@keras_export("keras.layers.deserialize")
def deserialize(config, custom_objects=None, use_legacy_format=False):
    """Instantiates a layer from a config dictionary.
    Args:
        config: dict of the form {'class_name': str, 'config': dict}
        custom_objects: dict mapping class names (or function names) of custom
          (non-Keras) objects to class/functions
    Returns:
        Layer instance (may be Model, Sequential, Network, Layer...)
    Example:
    ```python
    # Configuration of Dense(32, activation='relu')
    config = {
      'class_name': 'Dense',
      'config': {
        'activation': 'relu',
        'activity_regularizer': None,
        'bias_constraint': None,
        'bias_initializer': {'class_name': 'Zeros', 'config': {}},
        'bias_regularizer': None,
        'dtype': 'float32',
        'kernel_constraint': None,
        'kernel_initializer': {'class_name': 'GlorotUniform',
                               'config': {'seed': None}},
        'kernel_regularizer': None,
        'name': 'dense',
        'trainable': True,
        'units': 32,
        'use_bias': True
      }
    }
    dense_layer = tf.keras.layers.deserialize(config)
    ```
    """
    populate_deserializable_objects()
    if not config:
        raise ValueError(
            f"Cannot deserialize empty config. Received: config={config}"
        )
    if use_legacy_format:
        return legacy_serialization.deserialize_keras_object(
            config,
            module_objects=LOCAL.ALL_OBJECTS,
            custom_objects=custom_objects,
            printable_module_name="layer",
        )
    return serialization_lib.deserialize_keras_object(
        config,
        module_objects=LOCAL.ALL_OBJECTS,
        custom_objects=custom_objects,
        printable_module_name="layer",
    )
