@keras_export("keras.layers.UnitNormalization", v1=[])
class UnitNormalization(base_layer.Layer):
    """Unit normalization layer.
    Normalize a batch of inputs so that each input in the batch has a L2 norm
    equal to 1 (across the axes specified in `axis`).
    Example:
    >>> data = tf.constant(np.arange(6).reshape(2, 3), dtype=tf.float32)
    >>> normalized_data = tf.keras.layers.UnitNormalization()(data)
    >>> print(tf.reduce_sum(normalized_data[0, :] ** 2).numpy())
    1.0
    Args:
      axis: Integer or list/tuple. The axis or axes to normalize across.
        Typically this is the features axis or axes. The left-out axes are
        typically the batch axis or axes. Defaults to `-1`, the last dimension
        in the input.
    """
    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        if isinstance(axis, (list, tuple)):
            self.axis = list(axis)
        elif isinstance(axis, int):
            self.axis = axis
        else:
            raise TypeError(
                "Invalid value for `axis` argument: "
                "expected an int or a list/tuple of ints. "
                f"Received: axis={axis}"
            )
        self.supports_masking = True
    def build(self, input_shape):
        self.axis = tf_utils.validate_axis(self.axis, input_shape)
    def call(self, inputs):
        inputs = tf.cast(inputs, self.compute_dtype)
        return tf.linalg.l2_normalize(inputs, axis=self.axis)
    def compute_output_shape(self, input_shape):
        return input_shape
    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config
