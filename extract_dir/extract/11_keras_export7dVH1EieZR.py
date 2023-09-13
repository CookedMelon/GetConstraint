"/home/cc/Workspace/tfconstraint/keras/initializers/initializers.py"
@keras_export(
    "keras.initializers.LecunUniform", "keras.initializers.lecun_uniform", v1=[]
)
class LecunUniform(VarianceScaling):
    """Lecun uniform initializer.
     Also available via the shortcut function
    `tf.keras.initializers.lecun_uniform`.
    Draws samples from a uniform distribution within `[-limit, limit]`, where
    `limit = sqrt(3 / fan_in)` (`fan_in` is the number of input units in the
    weight tensor).
    Examples:
    >>> # Standalone usage:
    >>> initializer = tf.keras.initializers.LecunUniform()
    >>> values = initializer(shape=(2, 2))
    >>> # Usage in a Keras layer:
    >>> initializer = tf.keras.initializers.LecunUniform()
    >>> layer = tf.keras.layers.Dense(3, kernel_initializer=initializer)
    Args:
      seed: A Python integer. Used to make the behavior of the initializer
        deterministic. Note that a seeded initializer will not produce the same
        random values across multiple calls, but multiple initializers will
        produce the same sequence when constructed with the same seed value.
    References:
      - [Klambauer et al., 2017](https://arxiv.org/abs/1706.02515)
    """
    def __init__(self, seed=None):
        super().__init__(
            scale=1.0, mode="fan_in", distribution="uniform", seed=seed
        )
    def get_config(self):
        return {"seed": self.seed}
