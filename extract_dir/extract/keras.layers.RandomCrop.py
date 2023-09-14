@keras_export(
    "keras.layers.RandomCrop",
    "keras.layers.experimental.preprocessing.RandomCrop",
    v1=[],
)
class RandomCrop(base_layer.BaseRandomLayer):
    """A preprocessing layer which randomly crops images during training.
    During training, this layer will randomly choose a location to crop images
    down to a target size. The layer will crop all the images in the same batch
    to the same cropping location.
    At inference time, and during training if an input image is smaller than the
    target size, the input will be resized and cropped so as to return the
    largest possible window in the image that matches the target aspect ratio.
    If you need to apply random cropping at inference time, set `training` to
    True when calling the layer.
    Input pixel values can be of any range (e.g. `[0., 1.)` or `[0, 255]`) and
    of integer or floating point dtype. By default, the layer will output
    floats.
    For an overview and full list of preprocessing layers, see the preprocessing
    [guide](https://www.tensorflow.org/guide/keras/preprocessing_layers).
    Input shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `"channels_last"` format.
    Output shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., target_height, target_width, channels)`.
    Args:
        height: Integer, the height of the output shape.
        width: Integer, the width of the output shape.
        seed: Integer. Used to create a random seed.
    """
    def __init__(self, height, width, seed=None, **kwargs):
        base_preprocessing_layer.keras_kpl_gauge.get_cell("RandomCrop").set(
            True
        )
        super().__init__(
            **kwargs, autocast=False, seed=seed, force_generator=True
        )
        self.height = height
        self.width = width
        self.seed = seed
    def call(self, inputs, training=True):
        inputs = convert_inputs(inputs, dtype=self.compute_dtype)
        input_shape = tf.shape(inputs)
        h_diff = input_shape[H_AXIS] - self.height
        w_diff = input_shape[W_AXIS] - self.width
        def random_crop():
            dtype = input_shape.dtype
            rands = self._random_generator.random_uniform(
                [2], 0, dtype.max, dtype
            )
            h_start = rands[0] % (h_diff + 1)
            w_start = rands[1] % (w_diff + 1)
            return tf.image.crop_to_bounding_box(
                inputs, h_start, w_start, self.height, self.width
            )
        def resize():
            outputs = image_utils.smart_resize(
                inputs, [self.height, self.width]
            )
            # smart_resize will always output float32, so we need to re-cast.
            return tf.cast(outputs, self.compute_dtype)
        return tf.cond(
            tf.reduce_all((training, h_diff >= 0, w_diff >= 0)),
            random_crop,
            resize,
        )
    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        input_shape[H_AXIS] = self.height
        input_shape[W_AXIS] = self.width
        return tf.TensorShape(input_shape)
    def get_config(self):
        config = {
            "height": self.height,
            "width": self.width,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
