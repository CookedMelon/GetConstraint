@keras_export(
    "keras.initializers.Identity", "keras.initializers.identity", v1=[]
)
class Identity(Initializer):
    """Initializer that generates the identity matrix.
    Also available via the shortcut function `tf.keras.initializers.identity`.
    Only usable for generating 2D matrices.
    Examples:
    >>> # Standalone usage:
    >>> initializer = tf.keras.initializers.Identity()
    >>> values = initializer(shape=(2, 2))
    >>> # Usage in a Keras layer:
    >>> initializer = tf.keras.initializers.Identity()
    >>> layer = tf.keras.layers.Dense(3, kernel_initializer=initializer)
    Args:
      gain: Multiplicative factor to apply to the identity matrix.
    """
    def __init__(self, gain=1.0):
        self.gain = gain
    def __call__(self, shape, dtype=None, **kwargs):
        """Returns a tensor object initialized to a 2D identity matrix.
        Args:
          shape: Shape of the tensor. It should have exactly rank 2.
          dtype: Optional dtype of the tensor. Only floating point types are
           supported. If not specified, `tf.keras.backend.floatx()` is used,
           which default to `float32` unless you configured it otherwise
           (via `tf.keras.backend.set_floatx(float_dtype)`)
          **kwargs: Additional keyword arguments.
        """
        _validate_kwargs(
            self.__class__.__name__, kwargs, support_partition=False
        )
        dtype = _assert_float_dtype(_get_dtype(dtype))
        if len(shape) != 2:
            raise ValueError(
                "Identity matrix initializer can only be used for 2D matrices. "
                f"Received: shape={shape} of rank {len(shape)}."
            )
        layout = kwargs.pop("layout", None)
        if layout:
            return utils.call_with_layout(
                self._generate_init_val, layout, shape=shape, dtype=dtype
            )
        return self._generate_init_val(shape, dtype)
    def _generate_init_val(self, shape, dtype):
        initializer = tf.eye(*shape, dtype=dtype)
        return self.gain * initializer
    def get_config(self):
        return {"gain": self.gain}
