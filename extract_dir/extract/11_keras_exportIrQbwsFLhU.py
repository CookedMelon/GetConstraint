"/home/cc/Workspace/tfconstraint/keras/initializers/initializers.py"
@keras_export(
    "keras.initializers.GlorotNormal", "keras.initializers.glorot_normal", v1=[]
)
class GlorotNormal(VarianceScaling):
    """The Glorot normal initializer, also called Xavier normal initializer.
    Also available via the shortcut function
    `tf.keras.initializers.glorot_normal`.
    Draws samples from a truncated normal distribution centered on 0 with
    `stddev = sqrt(2 / (fan_in + fan_out))` where `fan_in` is the number of
    input units in the weight tensor and `fan_out` is the number of output units
    in the weight tensor.
    Examples:
    >>> # Standalone usage:
    >>> initializer = tf.keras.initializers.GlorotNormal()
    >>> values = initializer(shape=(2, 2))
    >>> # Usage in a Keras layer:
    >>> initializer = tf.keras.initializers.GlorotNormal()
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
            scale=1.0,
            mode="fan_avg",
            distribution="truncated_normal",
            seed=seed,
        )
    def get_config(self):
        return {"seed": self.seed}
