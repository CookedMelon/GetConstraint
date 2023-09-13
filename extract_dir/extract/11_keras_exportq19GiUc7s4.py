"/home/cc/Workspace/tfconstraint/keras/layers/preprocessing/image_preprocessing.py"
@keras_export(
    "keras.layers.CenterCrop",
    "keras.layers.experimental.preprocessing.CenterCrop",
)
class CenterCrop(base_layer.Layer):
    """A preprocessing layer which crops images.
    This layers crops the central portion of the images to a target size. If an
    image is smaller than the target size, it will be resized and cropped
    so as to return the largest possible window in the image that matches
    the target aspect ratio.
    Input pixel values can be of any range (e.g. `[0., 1.)` or `[0, 255]`) and
    of integer or floating point dtype.
    By default, the layer will output floats.
    For an overview and full list of preprocessing layers, see the preprocessing
    [guide](https://www.tensorflow.org/guide/keras/preprocessing_layers).
    Input shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `"channels_last"` format.
    Output shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., target_height, target_width, channels)`.
    If the input height/width is even and the target height/width is odd (or
    inversely), the input image is left-padded by 1 pixel.
    Args:
        height: Integer, the height of the output shape.
        width: Integer, the width of the output shape.
    """
    def __init__(self, height, width, **kwargs):
        self.height = height
        self.width = width
        super().__init__(**kwargs, autocast=False)
        base_preprocessing_layer.keras_kpl_gauge.get_cell("CenterCrop").set(
            True
        )
    def call(self, inputs):
        inputs = convert_inputs(inputs, self.compute_dtype)
        input_shape = tf.shape(inputs)
        h_diff = input_shape[H_AXIS] - self.height
        w_diff = input_shape[W_AXIS] - self.width
        def center_crop():
            h_start = tf.cast(h_diff / 2, tf.int32)
            w_start = tf.cast(w_diff / 2, tf.int32)
            return tf.image.crop_to_bounding_box(
                inputs, h_start, w_start, self.height, self.width
            )
        def upsize():
            outputs = image_utils.smart_resize(
                inputs, [self.height, self.width]
            )
            # smart_resize will always output float32, so we need to re-cast.
            return tf.cast(outputs, self.compute_dtype)
        return tf.cond(
            tf.reduce_all((h_diff >= 0, w_diff >= 0)), center_crop, upsize
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
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
