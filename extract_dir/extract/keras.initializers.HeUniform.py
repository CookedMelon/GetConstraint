@keras_export(
    "keras.initializers.HeUniform", "keras.initializers.he_uniform", v1=[]
)
class HeUniform(VarianceScaling):
    """He uniform variance scaling initializer.
     Also available via the shortcut function
    `tf.keras.initializers.he_uniform`.
    Draws samples from a uniform distribution within `[-limit, limit]`, where
    `limit = sqrt(6 / fan_in)` (`fan_in` is the number of input units in the
    weight tensor).
    Examples:
    >>> # Standalone usage:
    >>> initializer = tf.keras.initializers.HeUniform()
    >>> values = initializer(shape=(2, 2))
    >>> # Usage in a Keras layer:
    >>> initializer = tf.keras.initializers.HeUniform()
    >>> layer = tf.keras.layers.Dense(3, kernel_initializer=initializer)
    Args:
      seed: A Python integer. Used to make the behavior of the initializer
        deterministic. Note that a seeded initializer will not produce the same
        random values across multiple calls, but multiple initializers will
        produce the same sequence when constructed with the same seed value.
    References:
      - [He et al., 2015](https://arxiv.org/abs/1502.01852)
    """
    def __init__(self, seed=None):
        super().__init__(
            scale=2.0, mode="fan_in", distribution="uniform", seed=seed
        )
    def get_config(self):
        return {"seed": self.seed}
def _get_dtype(dtype):
    if dtype is None:
        dtype = backend.floatx()
    return tf.as_dtype(dtype)
def _assert_float_dtype(dtype):
    """Validate and return floating point type based on `dtype`.
    `dtype` must be a floating point type.
    Args:
      dtype: The data type to validate.
    Returns:
      Validated type.
    Raises:
      ValueError: if `dtype` is not a floating point type.
    """
    dtype = tf.as_dtype(dtype)
    if not dtype.is_floating:
        raise ValueError(f"Expected floating point type, got {dtype}.")
    return dtype
def _compute_fans(shape):
    """Computes the number of input and output units for a weight shape.
    Args:
      shape: Integer shape tuple or TF tensor shape.
    Returns:
      A tuple of integer scalars (fan_in, fan_out).
    """
    if len(shape) < 1:  # Just to avoid errors for constants.
        fan_in = fan_out = 1
    elif len(shape) == 1:
        fan_in = fan_out = shape[0]
    elif len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    else:
        # Assuming convolution kernels (2D, 3D, or more).
        # kernel shape: (..., input_depth, depth)
        receptive_field_size = 1
        for dim in shape[:-2]:
            receptive_field_size *= dim
        fan_in = shape[-2] * receptive_field_size
        fan_out = shape[-1] * receptive_field_size
    return int(fan_in), int(fan_out)
def _validate_kwargs(cls_name, kwargs, support_partition=True):
    invalid_kwargs = [k for k in kwargs if k not in _ALLOWED_INITIALIZER_KWARGS]
    if invalid_kwargs:
        raise TypeError(
            f"Unknown keyword arguments: {invalid_kwargs}. Allowed "
            f"keyword arguments: {_ALLOWED_INITIALIZER_KWARGS}."
        )
    if not support_partition and (
        _PARTITION_SHAPE in kwargs or _PARTITION_OFFSET in kwargs
    ):
        raise ValueError(
            f"{cls_name} initializer doesn't support "
            "partition-related arguments."
        )
def _ensure_keras_seeded():
    """Make sure the keras.backend global seed generator is set.
    This is important for DTensor use case to ensure that each client are
    initialized with same seed for tf.random.Generator, so that the value
    created are in sync among all the clients.
    """
    if not getattr(backend._SEED_GENERATOR, "generator", None):
        raise ValueError(
            "When using DTensor APIs, you need to set the global seed "
            "before using any Keras initializers. Please make sure "
            "to call `tf.keras.utils.set_random_seed()` in your code."
        )
