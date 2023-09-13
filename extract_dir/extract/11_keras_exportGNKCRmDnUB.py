"/home/cc/Workspace/tfconstraint/keras/layers/preprocessing/image_preprocessing.py"
@keras_export(
    "keras.layers.RandomWidth",
    "keras.layers.experimental.preprocessing.RandomWidth",
    v1=[],
)
class RandomWidth(base_layer.BaseRandomLayer):
    """A preprocessing layer which randomly varies image width during training.
    This layer will randomly adjusts the width of a batch of images of a
    batch of images by a random factor. The input should be a 3D (unbatched) or
    4D (batched) tensor in the `"channels_last"` image data format. Input pixel
    values can be of any range (e.g. `[0., 1.)` or `[0, 255]`) and of integer or
    floating point dtype. By default, the layer will output floats.
    By default, this layer is inactive during inference.
    For an overview and full list of preprocessing layers, see the preprocessing
    [guide](https://www.tensorflow.org/guide/keras/preprocessing_layers).
    Args:
        factor: A positive float (fraction of original width),
            or a tuple of size 2 representing lower and upper bound
            for resizing vertically. When represented as a single float,
            this value is used for both the upper and
            lower bound. For instance, `factor=(0.2, 0.3)`
            results in an output with
            width changed by a random amount in the range `[20%, 30%]`.
            `factor=(-0.2, 0.3)` results in an output with width changed
            by a random amount in the range `[-20%, +30%]`.
            `factor=0.2` results in an output with width changed
            by a random amount in the range `[-20%, +20%]`.
        interpolation: String, the interpolation method.
            Supports `"bilinear"`, `"nearest"`, `"bicubic"`, `"area"`,
            `"lanczos3"`, `"lanczos5"`, `"gaussian"`, `"mitchellcubic"`.
            Defaults to `bilinear`.
        seed: Integer. Used to create a random seed.
    Input shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `"channels_last"` format.
    Output shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, random_width, channels)`.
    """
    def __init__(self, factor, interpolation="bilinear", seed=None, **kwargs):
        base_preprocessing_layer.keras_kpl_gauge.get_cell("RandomWidth").set(
            True
        )
        super().__init__(seed=seed, force_generator=True, **kwargs)
        self.factor = factor
        if isinstance(factor, (tuple, list)):
            self.width_lower = factor[0]
            self.width_upper = factor[1]
        else:
            self.width_lower = -factor
            self.width_upper = factor
        if self.width_upper < self.width_lower:
            raise ValueError(
                "`factor` argument cannot have an upper bound less than the "
                f"lower bound. Received: factor={factor}"
            )
        if self.width_lower < -1.0 or self.width_upper < -1.0:
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
        def random_width_inputs(inputs):
            """Inputs width-adjusted with random ops."""
            inputs_shape = tf.shape(inputs)
            img_hd = inputs_shape[H_AXIS]
            img_wd = tf.cast(inputs_shape[W_AXIS], tf.float32)
            width_factor = self._random_generator.random_uniform(
                shape=[],
                minval=(1.0 + self.width_lower),
                maxval=(1.0 + self.width_upper),
            )
            adjusted_width = tf.cast(width_factor * img_wd, tf.int32)
            adjusted_size = tf.stack([img_hd, adjusted_width])
            output = tf.image.resize(
                images=inputs,
                size=adjusted_size,
                method=self._interpolation_method,
            )
            # tf.resize will output float32 regardless of input type.
            output = tf.cast(output, self.compute_dtype)
            output_shape = inputs.shape.as_list()
            output_shape[W_AXIS] = None
            output.set_shape(output_shape)
            return output
        if training:
            return random_width_inputs(inputs)
        else:
            return inputs
    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        input_shape[W_AXIS] = None
        return tf.TensorShape(input_shape)
    def get_config(self):
        config = {
            "factor": self.factor,
            "interpolation": self.interpolation,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
def convert_inputs(inputs, dtype=None):
    if isinstance(inputs, dict):
        raise ValueError(
            "This layer can only process a tensor representing an image or "
            f"a batch of images. Received: type(inputs)={type(inputs)}."
            "If you need to pass a dict containing "
            "images, labels, and bounding boxes, you should "
            "instead use the preprocessing and augmentation layers "
            "from `keras_cv.layers`. See docs at "
            "https://keras.io/api/keras_cv/layers/"
        )
    inputs = utils.ensure_tensor(inputs, dtype=dtype)
    return inputs
