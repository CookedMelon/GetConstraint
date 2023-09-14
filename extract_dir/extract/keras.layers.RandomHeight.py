@keras_export(
    "keras.layers.RandomHeight",
    "keras.layers.experimental.preprocessing.RandomHeight",
    v1=[],
)
class RandomHeight(base_layer.BaseRandomLayer):
    """A preprocessing layer which randomly varies image height during training.
    This layer adjusts the height of a batch of images by a random factor.
    The input should be a 3D (unbatched) or 4D (batched) tensor in the
    `"channels_last"` image data format. Input pixel values can be of any range
    (e.g. `[0., 1.)` or `[0, 255]`) and of integer or floating point dtype. By
    default, the layer will output floats.
    By default, this layer is inactive during inference.
    For an overview and full list of preprocessing layers, see the preprocessing
    [guide](https://www.tensorflow.org/guide/keras/preprocessing_layers).
    Args:
        factor: A positive float (fraction of original height),
            or a tuple of size 2 representing lower and upper bound
            for resizing vertically. When represented as a single float,
            this value is used for both the upper and
            lower bound. For instance, `factor=(0.2, 0.3)` results
            in an output with
            height changed by a random amount in the range `[20%, 30%]`.
            `factor=(-0.2, 0.3)` results in an output with height
            changed by a random amount in the range `[-20%, +30%]`.
            `factor=0.2` results in an output with
            height changed by a random amount in the range `[-20%, +20%]`.
        interpolation: String, the interpolation method.
            Supports `"bilinear"`, `"nearest"`, `"bicubic"`, `"area"`,
            `"lanczos3"`, `"lanczos5"`, `"gaussian"`, `"mitchellcubic"`.
            Defaults to `"bilinear"`.
        seed: Integer. Used to create a random seed.
    Input shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `"channels_last"` format.
    Output shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., random_height, width, channels)`.
    """
    def __init__(self, factor, interpolation="bilinear", seed=None, **kwargs):
        base_preprocessing_layer.keras_kpl_gauge.get_cell("RandomHeight").set(
            True
        )
        super().__init__(seed=seed, force_generator=True, **kwargs)
        self.factor = factor
        if isinstance(factor, (tuple, list)):
            self.height_lower = factor[0]
            self.height_upper = factor[1]
        else:
            self.height_lower = -factor
            self.height_upper = factor
        if self.height_upper < self.height_lower:
            raise ValueError(
                "`factor` argument cannot have an upper bound lesser than the "
                f"lower bound. Received: factor={factor}"
            )
        if self.height_lower < -1.0 or self.height_upper < -1.0:
            raise ValueError(
                "`factor` argument must have values larger than -1. "
                f"Received: factor={factor}"
            )
        self.interpolation = interpolation
        self._interpolation_method = image_utils.get_interpolation(
            interpolation
        )
        self.seed = seed
    def call(self, inputs, training=True):
        inputs = convert_inputs(inputs)
        def random_height_inputs(inputs):
            """Inputs height-adjusted with random ops."""
            inputs_shape = tf.shape(inputs)
            img_hd = tf.cast(inputs_shape[H_AXIS], tf.float32)
            img_wd = inputs_shape[W_AXIS]
            height_factor = self._random_generator.random_uniform(
                shape=[],
                minval=(1.0 + self.height_lower),
                maxval=(1.0 + self.height_upper),
            )
            adjusted_height = tf.cast(height_factor * img_hd, tf.int32)
            adjusted_size = tf.stack([adjusted_height, img_wd])
            output = tf.image.resize(
                images=inputs,
                size=adjusted_size,
                method=self._interpolation_method,
            )
            # tf.resize will output float32 regardless of input type.
            output = tf.cast(output, self.compute_dtype)
            output_shape = inputs.shape.as_list()
            output_shape[H_AXIS] = None
            output.set_shape(output_shape)
            return output
        if training:
            return random_height_inputs(inputs)
        else:
            return inputs
    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        input_shape[H_AXIS] = None
        return tf.TensorShape(input_shape)
    def get_config(self):
        config = {
            "factor": self.factor,
            "interpolation": self.interpolation,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
