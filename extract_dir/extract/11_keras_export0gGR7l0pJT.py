"/home/cc/Workspace/tfconstraint/keras/initializers/initializers.py"
@keras_export(
    "keras.initializers.GlorotUniform",
    "keras.initializers.glorot_uniform",
    v1=[],
)
class GlorotUniform(VarianceScaling):
    """The Glorot uniform initializer, also called Xavier uniform initializer.
    Also available via the shortcut function
    `tf.keras.initializers.glorot_uniform`.
    Draws samples from a uniform distribution within `[-limit, limit]`, where
    `limit = sqrt(6 / (fan_in + fan_out))` (`fan_in` is the number of input
    units in the weight tensor and `fan_out` is the number of output units).
    Examples:
    >>> # Standalone usage:
    >>> initializer = tf.keras.initializers.GlorotUniform()
    >>> values = initializer(shape=(2, 2))
    >>> # Usage in a Keras layer:
    >>> initializer = tf.keras.initializers.GlorotUniform()
    >>> layer = tf.keras.layers.Dense(3, kernel_initializer=initializer)
    Args:
      seed: A Python integer. Used to make the behavior of the initializer
        deterministic. Note that a seeded initializer will not produce the same
        random values across multiple calls, but multiple initializers will
        produce the same sequence when constructed with the same seed value.
    References:
      - [Glorot et al., 2010](http://proceedings.mlr.press/v9/glorot10a.html)
    """
    def __init__(self, seed=None):
        super().__init__(
            scale=1.0, mode="fan_avg", distribution="uniform", seed=seed
        )
    def get_config(self):
        return {"seed": self.seed}
