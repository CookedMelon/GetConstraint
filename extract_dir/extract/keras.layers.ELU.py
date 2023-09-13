@keras_export("keras.layers.ELU")
class ELU(Layer):
    """Exponential Linear Unit.
    It follows:
    ```
        f(x) =  alpha * (exp(x) - 1.) for x < 0
        f(x) = x for x >= 0
    ```
    Input shape:
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    Output shape:
        Same shape as the input.
    Args:
        alpha: Scale for the negative factor.
    """
    def __init__(self, alpha=1.0, **kwargs):
        super().__init__(**kwargs)
        if alpha is None:
            raise ValueError(
                "Alpha of an ELU layer cannot be None, expecting a float. "
                f"Received: {alpha}"
            )
        self.supports_masking = True
        self.alpha = backend.cast_to_floatx(alpha)
    def call(self, inputs):
        return backend.elu(inputs, self.alpha)
    def get_config(self):
        config = {"alpha": float(self.alpha)}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape
