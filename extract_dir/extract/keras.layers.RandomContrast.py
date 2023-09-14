@keras_export(
    "keras.layers.RandomContrast",
    "keras.layers.experimental.preprocessing.RandomContrast",
    v1=[],
)
class RandomContrast(base_layer.BaseRandomLayer):
    """A preprocessing layer which randomly adjusts contrast during training.
    This layer will randomly adjust the contrast of an image or images
    by a random factor. Contrast is adjusted independently
    for each channel of each image during training.
    For each channel, this layer computes the mean of the image pixels in the
    channel and then adjusts each component `x` of each pixel to
    `(x - mean) * contrast_factor + mean`.
    Input pixel values can be of any range (e.g. `[0., 1.)` or `[0, 255]`) and
    in integer or floating point dtype.
    By default, the layer will output floats.
    The output value will be clipped to the range `[0, 255]`, the valid
    range of RGB colors.
    For an overview and full list of preprocessing layers, see the preprocessing
    [guide](https://www.tensorflow.org/guide/keras/preprocessing_layers).
    Input shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `"channels_last"` format.
    Output shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `"channels_last"` format.
    Args:
        factor: a positive float represented as fraction of value, or a tuple of
            size 2 representing lower and upper bound.
            When represented as a single float, lower = upper.
            The contrast factor will be randomly picked between
            `[1.0 - lower, 1.0 + upper]`. For any pixel x in the channel,
            the output will be `(x - mean) * factor + mean`
            where `mean` is the mean value of the channel.
        seed: Integer. Used to create a random seed.
    """
    def __init__(self, factor, seed=None, **kwargs):
        base_preprocessing_layer.keras_kpl_gauge.get_cell("RandomContrast").set(
            True
        )
        super().__init__(seed=seed, force_generator=True, **kwargs)
        self.factor = factor
        if isinstance(factor, (tuple, list)):
            self.lower = factor[0]
            self.upper = factor[1]
        else:
            self.lower = self.upper = factor
        if self.lower < 0.0 or self.upper < 0.0 or self.lower > 1.0:
            raise ValueError(
                "`factor` argument cannot have negative values or values "
                "greater than 1."
                f"Received: factor={factor}"
            )
        self.seed = seed
    def call(self, inputs, training=True):
        inputs = convert_inputs(inputs, self.compute_dtype)
        def random_contrasted_inputs(inputs):
            seed = self._random_generator.make_seed_for_stateless_op()
            if seed is not None:
                output = tf.image.stateless_random_contrast(
                    inputs, 1.0 - self.lower, 1.0 + self.upper, seed=seed
                )
            else:
                output = tf.image.random_contrast(
                    inputs,
                    1.0 - self.lower,
                    1.0 + self.upper,
                    seed=self._random_generator.make_legacy_seed(),
                )
            output = tf.clip_by_value(output, 0, 255)
            output.set_shape(inputs.shape)
            return output
        if training:
            return random_contrasted_inputs(inputs)
        else:
            return inputs
    def compute_output_shape(self, input_shape):
        return input_shape
    def get_config(self):
        config = {
            "factor": self.factor,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
