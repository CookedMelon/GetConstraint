@keras_export(
    "keras.initializers.Constant", "keras.initializers.constant", v1=[]
)
class Constant(Initializer):
    """Initializer that generates tensors with constant values.
    Also available via the shortcut function `tf.keras.initializers.constant`.
    Only scalar values are allowed.
    The constant value provided must be convertible to the dtype requested
    when calling the initializer.
    Examples:
    >>> # Standalone usage:
    >>> initializer = tf.keras.initializers.Constant(3.)
    >>> values = initializer(shape=(2, 2))
    >>> # Usage in a Keras layer:
    >>> initializer = tf.keras.initializers.Constant(3.)
    >>> layer = tf.keras.layers.Dense(3, kernel_initializer=initializer)
    Args:
        value: A Python scalar.
    """
    def __init__(self, value=0):
        self.value = value
    def __call__(self, shape, dtype=None, **kwargs):
        """Returns a tensor object initialized to `self.value`.
        Args:
            shape: Shape of the tensor.
            dtype: Optional dtype of the tensor. If not specified,
                `keras.backend.floatx()` is used,
                which defaults to `float32` unless you configured it
                otherwise (via `keras.backend.set_floatx(float_dtype)`).
                **kwargs: Additional keyword arguments.
        """
        _validate_kwargs(self.__class__.__name__, kwargs)
        dtype = _get_dtype(dtype)
        if _PARTITION_SHAPE in kwargs:
            shape = kwargs[_PARTITION_SHAPE]
        layout = kwargs.pop("layout", None)
        if layout:
            return utils.call_with_layout(
                tf.constant, layout, self.value, shape=shape, dtype=dtype
            )
        return tf.constant(self.value, dtype=_get_dtype(dtype), shape=shape)
    def get_config(self):
        return {"value": self.value}
    @classmethod
    def from_config(cls, config):
        config.pop("dtype", None)
        if "value" in config:
            if isinstance(config["value"], dict):
                config["value"] = serialization_lib.deserialize_keras_object(
                    config["value"]
                )
        return cls(**config)
