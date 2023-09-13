"/home/cc/Workspace/tfconstraint/keras/initializers/initializers.py"
@keras_export(
    "keras.initializers.TruncatedNormal",
    "keras.initializers.truncated_normal",
    v1=[],
)
class TruncatedNormal(Initializer):
    """Initializer that generates a truncated normal distribution.
    Also available via the shortcut function
    `tf.keras.initializers.truncated_normal`.
    The values generated are similar to values from a
    `tf.keras.initializers.RandomNormal` initializer except that values more
    than two standard deviations from the mean are
    discarded and re-drawn.
    Examples:
    >>> # Standalone usage:
    >>> initializer = tf.keras.initializers.TruncatedNormal(mean=0., stddev=1.)
    >>> values = initializer(shape=(2, 2))
    >>> # Usage in a Keras layer:
    >>> initializer = tf.keras.initializers.TruncatedNormal(mean=0., stddev=1.)
    >>> layer = tf.keras.layers.Dense(3, kernel_initializer=initializer)
    Args:
      mean: a python scalar or a scalar tensor. Mean of the random values
        to generate.
      stddev: a python scalar or a scalar tensor. Standard deviation of the
        random values to generate before truncation.
      seed: A Python integer. Used to make the behavior of the initializer
        deterministic. Note that a seeded initializer will produce the same
        random values across multiple calls.
    """
    def __init__(self, mean=0.0, stddev=0.05, seed=None):
        self.mean = mean
        self.stddev = stddev
        self.seed = seed
        self._random_generator = backend.RandomGenerator(
            seed, rng_type="stateless"
        )
    def __call__(self, shape, dtype=None, **kwargs):
        """Returns a tensor initialized to random normal values (truncated).
        Args:
          shape: Shape of the tensor.
          dtype: Optional dtype of the tensor. Only floating point types are
            supported. If not specified, `tf.keras.backend.floatx()` is used,
            which default to `float32` unless you configured it otherwise (via
            `tf.keras.backend.set_floatx(float_dtype)`)
          **kwargs: Additional keyword arguments.
        """
        _validate_kwargs(self.__class__.__name__, kwargs)
        dtype = _assert_float_dtype(_get_dtype(dtype))
        if _PARTITION_SHAPE in kwargs:
            shape = kwargs[_PARTITION_SHAPE]
        partition_offset = kwargs.get(_PARTITION_OFFSET, None)
        if partition_offset is None:
            # We skip the reuse warning for partitioned variable, since the same
            # initializer will be called multiple times for each partition.
            self._warn_reuse()
        nonce = hash(partition_offset) if partition_offset else None
        layout = kwargs.pop("layout", None)
        if layout:
            # TODO(scottzhu): Remove this once the forward compat period above
            # is expired.
            self._random_generator._rng_type = (
                self._random_generator.RNG_STATEFUL
            )
            _ensure_keras_seeded()
            return utils.call_with_layout(
                self._random_generator.truncated_normal,
                layout,
                shape,
                self.mean,
                self.stddev,
                dtype,
                nonce,
            )
        return self._random_generator.truncated_normal(
            shape, self.mean, self.stddev, dtype, nonce
        )
    def get_config(self):
        return {"mean": self.mean, "stddev": self.stddev, "seed": self.seed}
