@keras_export("keras.optimizers.schedules.serialize")
def serialize(learning_rate_schedule, use_legacy_format=False):
    """Serializes a `LearningRateSchedule` into a JSON-compatible dict.
    Args:
      learning_rate_schedule: The `LearningRateSchedule` object to serialize.
    Returns:
      A JSON-serializable dict representing the object's config.
    Example:
    >>> lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    ...   0.1, decay_steps=100000, decay_rate=0.96, staircase=True)
    >>> tf.keras.optimizers.schedules.serialize(lr_schedule)
    {'module': 'keras.optimizers.schedules',
    'class_name': 'ExponentialDecay', 'config': {...},
    'registered_name': None}
    """
    if use_legacy_format:
        return legacy_serialization.serialize_keras_object(
            learning_rate_schedule
        )
    return serialization_lib.serialize_keras_object(learning_rate_schedule)
