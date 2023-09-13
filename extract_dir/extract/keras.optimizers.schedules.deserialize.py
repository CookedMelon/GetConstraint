@keras_export("keras.optimizers.schedules.deserialize")
def deserialize(config, custom_objects=None, use_legacy_format=False):
    """Instantiates a `LearningRateSchedule` object from a serialized form.
    Args:
      config: The serialized form of the `LearningRateSchedule`.
        Dictionary of the form {'class_name': str, 'config': dict}.
      custom_objects: A dictionary mapping class names (or function names) of
        custom (non-Keras) objects to class/functions.
    Returns:
      A `LearningRateSchedule` object.
    Example:
    ```python
    # Configuration for PolynomialDecay
    config = {
      'class_name': 'PolynomialDecay',
      'config': {'cycle': False,
        'decay_steps': 10000,
        'end_learning_rate': 0.01,
        'initial_learning_rate': 0.1,
        'name': None,
        'power': 0.5}}
    lr_schedule = tf.keras.optimizers.schedules.deserialize(config)
    ```
    """
    if use_legacy_format:
        return legacy_serialization.deserialize_keras_object(
            config,
            module_objects=globals(),
            custom_objects=custom_objects,
            printable_module_name="decay",
        )
    return serialization_lib.deserialize_keras_object(
        config,
        module_objects=globals(),
        custom_objects=custom_objects,
        printable_module_name="decay",
    )
