@keras_export("keras.backend.experimental.disable_tf_random_generator", v1=[])
def disable_tf_random_generator():
    """Disable the `tf.random.Generator` as the RNG for Keras.
    See `tf.keras.backend.experimental.is_tf_random_generator_enabled` for more
    details.
    """
    global _USE_GENERATOR_FOR_RNG
    _USE_GENERATOR_FOR_RNG = False
