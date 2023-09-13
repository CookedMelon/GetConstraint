"/home/cc/Workspace/tfconstraint/keras/initializers/initializers.py"
@keras_export(
    "keras.initializers.VarianceScaling",
    "keras.initializers.variance_scaling",
    v1=[],
)
class VarianceScaling(Initializer):
    """Initializer that adapts its scale to the shape of its input tensors.
    Also available via the shortcut function
    `tf.keras.initializers.variance_scaling`.
    With `distribution="truncated_normal" or "untruncated_normal"`, samples are
    drawn from a truncated/untruncated normal distribution with a mean of zero
    and a standard deviation (after truncation, if used) `stddev = sqrt(scale /
    n)`, where `n` is:
    - number of input units in the weight tensor, if `mode="fan_in"`
    - number of output units, if `mode="fan_out"`
    - average of the numbers of input and output units, if `mode="fan_avg"`
    With `distribution="uniform"`, samples are drawn from a uniform distribution
    within `[-limit, limit]`, where `limit = sqrt(3 * scale / n)`.
    Examples:
    >>> # Standalone usage:
    >>> initializer = tf.keras.initializers.VarianceScaling(
    ... scale=0.1, mode='fan_in', distribution='uniform')
    >>> values = initializer(shape=(2, 2))
    >>> # Usage in a Keras layer:
    >>> initializer = tf.keras.initializers.VarianceScaling(
    ... scale=0.1, mode='fan_in', distribution='uniform')
    >>> layer = tf.keras.layers.Dense(3, kernel_initializer=initializer)
    Args:
        scale: Scaling factor (positive float).
        mode: One of `"fan_in"`, `"fan_out"`, `"fan_avg"`.
        distribution: Random distribution to use. One of `"truncated_normal"`,
            `"untruncated_normal"`, or `"uniform"`.
        seed: A Python integer. Used to make the behavior of the initializer
            deterministic. Note that a seeded initializer will produce the same
            random values across multiple calls.
    """
    def __init__(
        self,
        scale=1.0,
        mode="fan_in",
        distribution="truncated_normal",
        seed=None,
    ):
        if scale <= 0.0:
            raise ValueError(
                f"`scale` must be positive float. Received: scale={scale}."
            )
        allowed_modes = {"fan_in", "fan_out", "fan_avg"}
        if mode not in allowed_modes:
            raise ValueError(
                f"Invalid `mode` argument: {mode}. "
                f"Please use one of the {allowed_modes}."
            )
        distribution = distribution.lower()
        # Compatibility with keras-team/keras.
        if distribution == "normal":
            distribution = "truncated_normal"
        allowed_distributions = {
            "uniform",
            "truncated_normal",
            "untruncated_normal",
        }
        if distribution not in allowed_distributions:
            raise ValueError(
                f"Invalid `distribution` argument: {distribution}."
                f"Allowed distributions: {allowed_distributions}."
            )
        self.scale = scale
        self.mode = mode
        self.distribution = distribution
        self.seed = seed
        self._random_generator = backend.RandomGenerator(
            seed, rng_type="stateless"
        )
    def __call__(self, shape, dtype=None, **kwargs):
        """Returns a tensor object initialized as specified by the initializer.
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
            _ensure_keras_seeded()
            return utils.call_with_layout(
                self._generate_init_val,
                layout,
                shape=shape,
                dtype=dtype,
                nonce=nonce,
            )
        return self._generate_init_val(shape=shape, dtype=dtype, nonce=nonce)
    def _generate_init_val(self, shape, dtype, nonce):
        scale = self.scale
        fan_in, fan_out = _compute_fans(shape)
        if self.mode == "fan_in":
            scale /= max(1.0, fan_in)
        elif self.mode == "fan_out":
            scale /= max(1.0, fan_out)
        else:
            scale /= max(1.0, (fan_in + fan_out) / 2.0)
        if self.distribution == "truncated_normal":
            # constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0.,
            # scale=1.)
            stddev = math.sqrt(scale) / 0.87962566103423978
            return self._random_generator.truncated_normal(
                shape, 0.0, stddev, dtype, nonce
            )
        elif self.distribution == "untruncated_normal":
            stddev = math.sqrt(scale)
            return self._random_generator.random_normal(
                shape, 0.0, stddev, dtype, nonce
            )
        else:
            limit = math.sqrt(3.0 * scale)
            return self._random_generator.random_uniform(
                shape, -limit, limit, dtype, nonce
            )
    def get_config(self):
        return {
            "scale": self.scale,
            "mode": self.mode,
            "distribution": self.distribution,
            "seed": self.seed,
        }
