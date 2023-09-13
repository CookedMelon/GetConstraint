"/home/cc/Workspace/tfconstraint/keras/initializers/initializers.py"
@keras_export(
    "keras.initializers.LecunNormal", "keras.initializers.lecun_normal", v1=[]
)
class LecunNormal(VarianceScaling):
    """Lecun normal initializer.
     Also available via the shortcut function
    `tf.keras.initializers.lecun_normal`.
    Initializers allow you to pre-specify an initialization strategy, encoded in
    the Initializer object, without knowing the shape and dtype of the variable
    being initialized.
    Draws samples from a truncated normal distribution centered on 0 with
    `stddev = sqrt(1 / fan_in)` where `fan_in` is the number of input units in
    the weight tensor.
    Examples:
    >>> # Standalone usage:
    >>> initializer = tf.keras.initializers.LecunNormal()
    >>> values = initializer(shape=(2, 2))
    >>> # Usage in a Keras layer:
    >>> initializer = tf.keras.initializers.LecunNormal()
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
            scale=1.0, mode="fan_in", distribution="truncated_normal", seed=seed
        )
    def get_config(self):
        return {"seed": self.seed}
