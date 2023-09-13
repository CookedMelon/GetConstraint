@keras_export("keras.layers.ThresholdedReLU")
class ThresholdedReLU(Layer):
    """Thresholded Rectified Linear Unit.
    It follows:
    ```
        f(x) = x for x > theta
        f(x) = 0 otherwise`
    ```
    Input shape:
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    Output shape:
        Same shape as the input.
    Args:
        theta: Float >= 0. Threshold location of activation.
    """
    def __init__(self, theta=1.0, **kwargs):
        super().__init__(**kwargs)
        if theta is None:
            raise ValueError(
                "Theta of a Thresholded ReLU layer cannot be None, expecting a "
                f"float. Received: {theta}"
            )
        if theta < 0:
            raise ValueError(
                "The theta value of a Thresholded ReLU layer "
                f"should be >=0. Received: {theta}"
            )
        self.supports_masking = True
        self.theta = backend.cast_to_floatx(theta)
    def call(self, inputs):
        dtype = self.compute_dtype
        return inputs * tf.cast(tf.greater(inputs, self.theta), dtype)
    def get_config(self):
        config = {"theta": float(self.theta)}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape
