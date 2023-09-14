@keras_export(
    "keras.layers.Resizing", "keras.layers.experimental.preprocessing.Resizing"
)
class Resizing(base_layer.Layer):
    """A preprocessing layer which resizes images.
    This layer resizes an image input to a target height and width. The input
    should be a 4D (batched) or 3D (unbatched) tensor in `"channels_last"`
    format. Input pixel values can be of any range
    (e.g. `[0., 1.)` or `[0, 255]`) and of integer or floating point dtype.
    By default, the layer will output floats.
    This layer can be called on tf.RaggedTensor batches of input images of
    distinct sizes, and will resize the outputs to dense tensors of uniform
    size.
    For an overview and full list of preprocessing layers, see the preprocessing
    [guide](https://www.tensorflow.org/guide/keras/preprocessing_layers).
    Args:
        height: Integer, the height of the output shape.
        width: Integer, the width of the output shape.
        interpolation: String, the interpolation method.
            Supports `"bilinear"`, `"nearest"`, `"bicubic"`, `"area"`,
            `"lanczos3"`, `"lanczos5"`, `"gaussian"`, `"mitchellcubic"`.
            Defaults to `"bilinear"`.
        crop_to_aspect_ratio: If True, resize the images without aspect
            ratio distortion. When the original aspect ratio differs
            from the target aspect ratio, the output image will be
            cropped so as to return the
            largest possible window in the image (of size `(height, width)`)
            that matches the target aspect ratio. By default
            (`crop_to_aspect_ratio=False`), aspect ratio may not be preserved.
    """
    def __init__(
        self,
        height,
        width,
        interpolation="bilinear",
        crop_to_aspect_ratio=False,
        **kwargs,
    ):
        self.height = height
        self.width = width
        self.interpolation = interpolation
        self.crop_to_aspect_ratio = crop_to_aspect_ratio
        self._interpolation_method = image_utils.get_interpolation(
            interpolation
        )
        super().__init__(**kwargs)
        base_preprocessing_layer.keras_kpl_gauge.get_cell("Resizing").set(True)
    def call(self, inputs):
        # tf.image.resize will always output float32
        # and operate more efficiently on float32
        # unless interpolation is nearest, in which case ouput type matches
        # input type.
        if self.interpolation == "nearest":
            input_dtype = self.compute_dtype
        else:
            input_dtype = tf.float32
        inputs = convert_inputs(inputs, dtype=input_dtype)
        size = [self.height, self.width]
        if self.crop_to_aspect_ratio:
            def resize_to_aspect(x):
                if tf_utils.is_ragged(inputs):
                    x = x.to_tensor()
                return image_utils.smart_resize(
                    x, size=size, interpolation=self._interpolation_method
                )
            if tf_utils.is_ragged(inputs):
                size_as_shape = tf.TensorShape(size)
                shape = size_as_shape + inputs.shape[-1:]
                spec = tf.TensorSpec(shape, input_dtype)
                outputs = tf.map_fn(
                    resize_to_aspect, inputs, fn_output_signature=spec
                )
            else:
                outputs = resize_to_aspect(inputs)
        else:
            outputs = tf.image.resize(
                inputs, size=size, method=self._interpolation_method
            )
        return tf.cast(outputs, self.compute_dtype)
    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        input_shape[H_AXIS] = self.height
        input_shape[W_AXIS] = self.width
        return tf.TensorShape(input_shape)
    def get_config(self):
        config = {
            "height": self.height,
            "width": self.width,
            "interpolation": self.interpolation,
            "crop_to_aspect_ratio": self.crop_to_aspect_ratio,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
