@keras_export("keras.layers.GaussianDropout")
class GaussianDropout(base_layer.BaseRandomLayer):
    """Apply multiplicative 1-centered Gaussian noise.
    As it is a regularization layer, it is only active at training time.
    Args:
      rate: Float, drop probability (as with `Dropout`).
        The multiplicative noise will have
        standard deviation `sqrt(rate / (1 - rate))`.
      seed: Integer, optional random seed to enable deterministic behavior.
    Call arguments:
      inputs: Input tensor (of any rank).
      training: Python boolean indicating whether the layer should behave in
        training mode (adding dropout) or in inference mode (doing nothing).
    Input shape:
      Arbitrary. Use the keyword argument `input_shape`
      (tuple of integers, does not include the samples axis)
      when using this layer as the first layer in a model.
    Output shape:
      Same shape as input.
    """
    def __init__(self, rate, seed=None, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self.supports_masking = True
        self.rate = rate
        self.seed = seed
    def call(self, inputs, training=None):
        if 0 < self.rate < 1:
            def noised():
                stddev = np.sqrt(self.rate / (1.0 - self.rate))
                return inputs * self._random_generator.random_normal(
                    shape=tf.shape(inputs),
                    mean=1.0,
                    stddev=stddev,
                    dtype=inputs.dtype,
                )
            return backend.in_train_phase(noised, inputs, training=training)
        return inputs
    def get_config(self):
        config = {"rate": self.rate, "seed": self.seed}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape
