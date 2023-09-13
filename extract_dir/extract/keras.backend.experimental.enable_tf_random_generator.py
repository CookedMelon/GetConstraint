@keras_export("keras.backend.experimental.enable_tf_random_generator", v1=[])
def enable_tf_random_generator():
    """Enable the `tf.random.Generator` as the RNG for Keras.
    See `tf.keras.backend.experimental.is_tf_random_generator_enabled` for more
    details.
    """
    global _USE_GENERATOR_FOR_RNG
    _USE_GENERATOR_FOR_RNG = True
