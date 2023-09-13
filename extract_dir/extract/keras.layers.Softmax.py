@keras_export("keras.layers.Softmax")
class Softmax(Layer):
    """Softmax activation function.
    Example without mask:
    >>> inp = np.asarray([1., 2., 1.])
    >>> layer = tf.keras.layers.Softmax()
    >>> layer(inp).numpy()
    array([0.21194157, 0.5761169 , 0.21194157], dtype=float32)
    >>> mask = np.asarray([True, False, True], dtype=bool)
    >>> layer(inp, mask).numpy()
    array([0.5, 0. , 0.5], dtype=float32)
    Input shape:
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    Output shape:
        Same shape as the input.
    Args:
        axis: Integer, or list of Integers, axis along which the softmax
            normalization is applied.
    Call arguments:
        inputs: The inputs, or logits to the softmax layer.
        mask: A boolean mask of the same shape as `inputs`. The mask
            specifies 1 to keep and 0 to mask. Defaults to `None`.
    Returns:
        Softmaxed output with the same shape as `inputs`.
    """
    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
    def call(self, inputs, mask=None):
        if mask is not None:
            # Since mask is 1.0 for positions we want to keep and 0.0 for masked
            # positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -1e.9 for masked positions.
            adder = (1.0 - tf.cast(mask, inputs.dtype)) * (
                _large_compatible_negative(inputs.dtype)
            )
            # Since we are adding it to the raw scores before the softmax, this
            # is effectively the same as removing these entirely.
            inputs += adder
        if isinstance(self.axis, (tuple, list)):
            if len(self.axis) > 1:
                return tf.exp(
                    inputs
                    - tf.reduce_logsumexp(inputs, axis=self.axis, keepdims=True)
                )
            else:
                return backend.softmax(inputs, axis=self.axis[0])
        return backend.softmax(inputs, axis=self.axis)
    def get_config(self):
        config = {"axis": self.axis}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape
