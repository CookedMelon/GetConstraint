"/home/cc/Workspace/tfconstraint/keras/initializers/initializers.py"
@keras_export(
    "keras.initializers.RandomUniform",
    "keras.initializers.random_uniform",
    v1=[],
)
class RandomUniform(Initializer):
    """Initializer that generates tensors with a uniform distribution.
    Also available via the shortcut function
    `tf.keras.initializers.random_uniform`.
    Examples:
    >>> # Standalone usage:
    >>> initializer = tf.keras.initializers.RandomUniform(minval=0., maxval=1.)
    >>> values = initializer(shape=(2, 2))
    >>> # Usage in a Keras layer:
    >>> initializer = tf.keras.initializers.RandomUniform(minval=0., maxval=1.)
    >>> layer = tf.keras.layers.Dense(3, kernel_initializer=initializer)
    Args:
      minval: A python scalar or a scalar tensor. Lower bound of the range of
        random values to generate (inclusive).
      maxval: A python scalar or a scalar tensor. Upper bound of the range of
        random values to generate (exclusive).
      seed: A Python integer. Used to make the behavior of the initializer
        deterministic. Note that a seeded initializer will produce the same
        random values across multiple calls.
    """
    def __init__(self, minval=-0.05, maxval=0.05, seed=None):
        self.minval = minval
        self.maxval = maxval
        self.seed = seed
        self._random_generator = backend.RandomGenerator(
            seed, rng_type="stateless"
        )
    def __call__(self, shape, dtype=None, **kwargs):
        """Returns a tensor object initialized as specified by the initializer.
        Args:
          shape: Shape of the tensor.
          dtype: Optional dtype of the tensor. Only floating point and integer
          types are supported. If not specified,
            `tf.keras.backend.floatx()` is used,
           which default to `float32` unless you configured it otherwise
           (via `tf.keras.backend.set_floatx(float_dtype)`).
          **kwargs: Additional keyword arguments.
        """
        _validate_kwargs(self.__class__.__name__, kwargs)
        dtype = _get_dtype(dtype)
        if not dtype.is_floating and not dtype.is_integer:
            raise ValueError(f"Expected float or integer dtype, got {dtype}.")
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
            _ensure_keras_seeded()
            return utils.call_with_layout(
                self._random_generator.random_uniform,
                layout,
                shape,
                self.minval,
                self.maxval,
                dtype,
                nonce,
            )
        return self._random_generator.random_uniform(
            shape, self.minval, self.maxval, dtype, nonce
        )
    def get_config(self):
        return {"minval": self.minval, "maxval": self.maxval, "seed": self.seed}
