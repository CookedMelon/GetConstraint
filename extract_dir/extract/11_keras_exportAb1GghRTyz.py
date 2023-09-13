"/home/cc/Workspace/tfconstraint/keras/layers/preprocessing/image_preprocessing.py"
@keras_export(
    "keras.layers.Rescaling",
    "keras.layers.experimental.preprocessing.Rescaling",
)
class Rescaling(base_layer.Layer):
    """A preprocessing layer which rescales input values to a new range.
    This layer rescales every value of an input (often an image) by multiplying
    by `scale` and adding `offset`.
    For instance:
    1. To rescale an input in the `[0, 255]` range
    to be in the `[0, 1]` range, you would pass `scale=1./255`.
    2. To rescale an input in the `[0, 255]` range to be in the `[-1, 1]` range,
    you would pass `scale=1./127.5, offset=-1`.
    The rescaling is applied both during training and inference. Inputs can be
    of integer or floating point dtype, and by default the layer will output
    floats.
    For an overview and full list of preprocessing layers, see the preprocessing
    [guide](https://www.tensorflow.org/guide/keras/preprocessing_layers).
    Input shape:
        Arbitrary.
    Output shape:
        Same as input.
    Args:
        scale: Float, the scale to apply to the inputs.
        offset: Float, the offset to apply to the inputs.
    """
    def __init__(self, scale, offset=0.0, **kwargs):
        self.scale = scale
        self.offset = offset
        super().__init__(**kwargs)
        base_preprocessing_layer.keras_kpl_gauge.get_cell("Rescaling").set(True)
    def call(self, inputs):
        dtype = self.compute_dtype
        inputs = convert_inputs(inputs, dtype=dtype)
        scale = tf.cast(self.scale, dtype)
        offset = tf.cast(self.offset, dtype)
        return tf.cast(inputs, dtype) * scale + offset
    def compute_output_shape(self, input_shape):
        return input_shape
    def get_config(self):
        config = {
            "scale": self.scale,
            "offset": self.offset,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
HORIZONTAL = "horizontal"
VERTICAL = "vertical"
HORIZONTAL_AND_VERTICAL = "horizontal_and_vertical"
