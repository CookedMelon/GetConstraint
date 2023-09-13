@keras_export("keras.initializers.Zeros", "keras.initializers.zeros", v1=[])
class Zeros(Initializer):
    """Initializer that generates tensors initialized to 0.
    Also available via the shortcut function `tf.keras.initializers.zeros`.
    Examples:
    >>> # Standalone usage:
    >>> initializer = tf.keras.initializers.Zeros()
    >>> values = initializer(shape=(2, 2))
    >>> # Usage in a Keras layer:
    >>> initializer = tf.keras.initializers.Zeros()
    >>> layer = tf.keras.layers.Dense(3, kernel_initializer=initializer)
    """
    def __call__(self, shape, dtype=None, **kwargs):
        """Returns a tensor object initialized as specified by the initializer.
        Args:
            shape: Shape of the tensor.
            dtype: Optional dtype of the tensor. Only numeric or boolean dtypes
                are supported. If not specified, `keras.backend.floatx()` is
                used, which defaults to `float32` unless you configured it
                otherwise (via `keras.backend.set_floatx(float_dtype)`).
            **kwargs: Additional keyword arguments.
        """
        _validate_kwargs(self.__class__.__name__, kwargs)
        dtype = _get_dtype(dtype)
        if not dtype.is_numpy_compatible or dtype == tf.string:
            raise ValueError(f"Expected numeric or boolean dtype, got {dtype}.")
        if _PARTITION_SHAPE in kwargs:
            shape = kwargs[_PARTITION_SHAPE]
        layout = kwargs.pop("layout", None)
        if layout:
            return utils.call_with_layout(
                tf.zeros, layout, shape=shape, dtype=dtype
            )
        return tf.zeros(shape, dtype)
