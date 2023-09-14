@keras_export(
    "keras.backend.experimental.is_tf_random_generator_enabled", v1=[]
)
def is_tf_random_generator_enabled():
    """Check whether `tf.random.Generator` is used for RNG in Keras.
    Compared to existing TF stateful random ops, `tf.random.Generator` uses
    `tf.Variable` and stateless random ops to generate random numbers,
    which leads to better reproducibility in distributed training.
    Note enabling it might introduce some breakage to existing code,
    by producing differently-seeded random number sequences
    and breaking tests that rely on specific random numbers being generated.
    To disable the
    usage of `tf.random.Generator`, please use
    `tf.keras.backend.experimental.disable_random_generator`.
    We expect the `tf.random.Generator` code path to become the default, and
    will remove the legacy stateful random ops such as `tf.random.uniform` in
    the future (see the [TF RNG guide](
    https://www.tensorflow.org/guide/random_numbers)).
    This API will also be removed in a future release as well, together with
    `tf.keras.backend.experimental.enable_tf_random_generator()` and
    `tf.keras.backend.experimental.disable_tf_random_generator()`
    Returns:
      boolean: whether `tf.random.Generator` is used for random number
        generation in Keras.
    """
    return _USE_GENERATOR_FOR_RNG
