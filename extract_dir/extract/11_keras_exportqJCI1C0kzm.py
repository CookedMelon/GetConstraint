"/home/cc/Workspace/tfconstraint/keras/layers/preprocessing/image_preprocessing.py"
@keras_export(
    "keras.layers.RandomFlip",
    "keras.layers.experimental.preprocessing.RandomFlip",
    v1=[],
)
class RandomFlip(base_layer.BaseRandomLayer):
    """A preprocessing layer which randomly flips images during training.
    This layer will flip the images horizontally and or vertically based on the
    `mode` attribute. During inference time, the output will be identical to
    input. Call the layer with `training=True` to flip the input.
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
        `(..., height, width, channels)`, in `"channels_last"` format.
    Args:
        mode: String indicating which flip mode to use. Can be `"horizontal"`,
            `"vertical"`, or `"horizontal_and_vertical"`. `"horizontal"` is a
            left-right flip and `"vertical"` is a top-bottom flip. Defaults to
            `"horizontal_and_vertical"`
        seed: Integer. Used to create a random seed.
    """
    def __init__(self, mode=HORIZONTAL_AND_VERTICAL, seed=None, **kwargs):
        super().__init__(seed=seed, force_generator=True, **kwargs)
        base_preprocessing_layer.keras_kpl_gauge.get_cell("RandomFlip").set(
            True
        )
        self.mode = mode
        if mode == HORIZONTAL:
            self.horizontal = True
            self.vertical = False
        elif mode == VERTICAL:
            self.horizontal = False
            self.vertical = True
        elif mode == HORIZONTAL_AND_VERTICAL:
            self.horizontal = True
            self.vertical = True
        else:
            raise ValueError(
                f"RandomFlip layer {self.name} received an unknown mode "
                f"argument {mode}"
            )
        self.seed = seed
    def call(self, inputs, training=True):
        inputs = convert_inputs(inputs, self.compute_dtype)
        def random_flipped_inputs(inputs):
            flipped_outputs = inputs
            if self.horizontal:
                seed = self._random_generator.make_seed_for_stateless_op()
                if seed is not None:
                    flipped_outputs = tf.image.stateless_random_flip_left_right(
                        flipped_outputs, seed=seed
                    )
                else:
                    flipped_outputs = tf.image.random_flip_left_right(
                        flipped_outputs,
                        self._random_generator.make_legacy_seed(),
                    )
            if self.vertical:
                seed = self._random_generator.make_seed_for_stateless_op()
                if seed is not None:
                    flipped_outputs = tf.image.stateless_random_flip_up_down(
                        flipped_outputs, seed=seed
                    )
                else:
                    flipped_outputs = tf.image.random_flip_up_down(
                        flipped_outputs,
                        self._random_generator.make_legacy_seed(),
                    )
            flipped_outputs.set_shape(inputs.shape)
            return flipped_outputs
        if training:
            return random_flipped_inputs(inputs)
        else:
            return inputs
    def compute_output_shape(self, input_shape):
        return input_shape
    def get_config(self):
        config = {
            "mode": self.mode,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
# TODO(tanzheny): Add examples, here and everywhere.
