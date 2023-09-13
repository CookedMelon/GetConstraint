"/home/cc/Workspace/tfconstraint/keras/layers/preprocessing/image_preprocessing.py"
@keras_export(
    "keras.layers.RandomTranslation",
    "keras.layers.experimental.preprocessing.RandomTranslation",
    v1=[],
)
class RandomTranslation(base_layer.BaseRandomLayer):
    """A preprocessing layer which randomly translates images during training.
    This layer will apply random translations to each image during training,
    filling empty space according to `fill_mode`.
    Input pixel values can be of any range (e.g. `[0., 1.)` or `[0, 255]`) and
    of integer or floating point dtype. By default, the layer will output
    floats.
    For an overview and full list of preprocessing layers, see the preprocessing
    [guide](https://www.tensorflow.org/guide/keras/preprocessing_layers).
    Args:
      height_factor: a float represented as fraction of value, or a tuple of
          size 2 representing lower and upper bound for shifting vertically. A
          negative value means shifting image up, while a positive value means
          shifting image down. When represented as a single positive float, this
          value is used for both the upper and lower bound. For instance,
          `height_factor=(-0.2, 0.3)` results in an output shifted by a random
          amount in the range `[-20%, +30%]`.  `height_factor=0.2` results in an
          output height shifted by a random amount in the range `[-20%, +20%]`.
      width_factor: a float represented as fraction of value, or a tuple of size
          2 representing lower and upper bound for shifting horizontally. A
          negative value means shifting image left, while a positive value means
          shifting image right. When represented as a single positive float,
          this value is used for both the upper and lower bound. For instance,
          `width_factor=(-0.2, 0.3)` results in an output shifted left by 20%,
          and shifted right by 30%. `width_factor=0.2` results
          in an output height shifted left or right by 20%.
      fill_mode: Points outside the boundaries of the input are filled according
          to the given mode
          (one of `{"constant", "reflect", "wrap", "nearest"}`).
          - *reflect*: `(d c b a | a b c d | d c b a)` The input is extended by
              reflecting about the edge of the last pixel.
          - *constant*: `(k k k k | a b c d | k k k k)` The input is extended by
              filling all values beyond the edge with the same constant value
              k = 0.
          - *wrap*: `(a b c d | a b c d | a b c d)` The input is extended by
              wrapping around to the opposite edge.
          - *nearest*: `(a a a a | a b c d | d d d d)` The input is extended by
              the nearest pixel.
      interpolation: Interpolation mode. Supported values: `"nearest"`,
          `"bilinear"`.
      seed: Integer. Used to create a random seed.
      fill_value: a float represents the value to be filled outside the
          boundaries when `fill_mode="constant"`.
    Input shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`,  in `"channels_last"` format.
    Output shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`,  in `"channels_last"` format.
    """
    def __init__(
        self,
        height_factor,
        width_factor,
        fill_mode="reflect",
        interpolation="bilinear",
        seed=None,
        fill_value=0.0,
        **kwargs,
    ):
        base_preprocessing_layer.keras_kpl_gauge.get_cell(
            "RandomTranslation"
        ).set(True)
        super().__init__(seed=seed, force_generator=True, **kwargs)
        self.height_factor = height_factor
        if isinstance(height_factor, (tuple, list)):
            self.height_lower = height_factor[0]
            self.height_upper = height_factor[1]
        else:
            self.height_lower = -height_factor
            self.height_upper = height_factor
        if self.height_upper < self.height_lower:
            raise ValueError(
                "`height_factor` cannot have upper bound less than "
                f"lower bound, got {height_factor}"
            )
        if abs(self.height_lower) > 1.0 or abs(self.height_upper) > 1.0:
            raise ValueError(
                "`height_factor` argument must have values between [-1, 1]. "
                f"Received: height_factor={height_factor}"
            )
        self.width_factor = width_factor
        if isinstance(width_factor, (tuple, list)):
            self.width_lower = width_factor[0]
            self.width_upper = width_factor[1]
        else:
            self.width_lower = -width_factor
            self.width_upper = width_factor
        if self.width_upper < self.width_lower:
            raise ValueError(
                "`width_factor` cannot have upper bound less than "
                f"lower bound, got {width_factor}"
            )
        if abs(self.width_lower) > 1.0 or abs(self.width_upper) > 1.0:
            raise ValueError(
                "`width_factor` must have values between [-1, 1], "
                f"got {width_factor}"
            )
        check_fill_mode_and_interpolation(fill_mode, interpolation)
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.interpolation = interpolation
        self.seed = seed
    def call(self, inputs, training=True):
        inputs = convert_inputs(inputs, self.compute_dtype)
        def random_translated_inputs(inputs):
            """Translated inputs with random ops."""
            # The transform op only accepts rank 4 inputs,
            # so if we have an unbatched image,
            # we need to temporarily expand dims to a batch.
            original_shape = inputs.shape
            unbatched = inputs.shape.rank == 3
            if unbatched:
                inputs = tf.expand_dims(inputs, 0)
            inputs_shape = tf.shape(inputs)
            batch_size = inputs_shape[0]
            img_hd = tf.cast(inputs_shape[H_AXIS], tf.float32)
            img_wd = tf.cast(inputs_shape[W_AXIS], tf.float32)
            height_translate = self._random_generator.random_uniform(
                shape=[batch_size, 1],
                minval=self.height_lower,
                maxval=self.height_upper,
                dtype=tf.float32,
            )
            height_translate = height_translate * img_hd
            width_translate = self._random_generator.random_uniform(
                shape=[batch_size, 1],
                minval=self.width_lower,
                maxval=self.width_upper,
                dtype=tf.float32,
            )
            width_translate = width_translate * img_wd
            translations = tf.cast(
                tf.concat([width_translate, height_translate], axis=1),
                dtype=tf.float32,
            )
            output = transform(
                inputs,
                get_translation_matrix(translations),
                interpolation=self.interpolation,
                fill_mode=self.fill_mode,
                fill_value=self.fill_value,
            )
            if unbatched:
                output = tf.squeeze(output, 0)
            output.set_shape(original_shape)
            return output
        if training:
            return random_translated_inputs(inputs)
        else:
            return inputs
    def compute_output_shape(self, input_shape):
        return input_shape
    def get_config(self):
        config = {
            "height_factor": self.height_factor,
            "width_factor": self.width_factor,
            "fill_mode": self.fill_mode,
            "fill_value": self.fill_value,
            "interpolation": self.interpolation,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
def get_translation_matrix(translations, name=None):
    """Returns projective transform(s) for the given translation(s).
    Args:
        translations: A matrix of 2-element lists representing `[dx, dy]`
            to translate for each image (for a batch of images).
        name: The name of the op.
    Returns:
        A tensor of shape `(num_images, 8)` projective transforms
            which can be given to `transform`.
    """
    with backend.name_scope(name or "translation_matrix"):
        num_translations = tf.shape(translations)[0]
        # The translation matrix looks like:
        #     [[1 0 -dx]
        #      [0 1 -dy]
        #      [0 0 1]]
        # where the last entry is implicit.
        # Translation matrices are always float32.
        return tf.concat(
            values=[
                tf.ones((num_translations, 1), tf.float32),
                tf.zeros((num_translations, 1), tf.float32),
                -translations[:, 0, None],
                tf.zeros((num_translations, 1), tf.float32),
                tf.ones((num_translations, 1), tf.float32),
                -translations[:, 1, None],
                tf.zeros((num_translations, 2), tf.float32),
            ],
            axis=1,
        )
def transform(
    images,
    transforms,
    fill_mode="reflect",
    fill_value=0.0,
    interpolation="bilinear",
    output_shape=None,
    name=None,
):
    """Applies the given transform(s) to the image(s).
    Args:
        images: A tensor of shape
            `(num_images, num_rows, num_columns, num_channels)` (NHWC).
            The rank must be statically known
            (the shape is not `TensorShape(None)`).
        transforms: Projective transform matrix/matrices.
            A vector of length 8 or tensor of size N x 8.
            If one row of transforms is [a0, a1, a2, b0, b1, b2,
            c0, c1], then it maps the *output* point `(x, y)`
            to a transformed *input* point
            `(x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k)`, where
            `k = c0 x + c1 y + 1`. The transforms are *inverted* compared to the
            transform mapping input points to output points.
            Note that gradients are not backpropagated
            into transformation parameters.
        fill_mode: Points outside the boundaries of the input are filled
            according to the given mode
            (one of `{"constant", "reflect", "wrap", "nearest"}`).
        fill_value: a float represents the value to be filled outside
            the boundaries when `fill_mode="constant"`.
        interpolation: Interpolation mode. Supported values: `"nearest"`,
            `"bilinear"`.
        output_shape: Output dimension after the transform, `[height, width]`.
            If `None`, output is the same size as input image.
        name: The name of the op.
    Fill mode behavior for each valid value is as follows:
    - `"reflect"`: `(d c b a | a b c d | d c b a)`
    The input is extended by reflecting about the edge of the last pixel.
    - `"constant"`: `(k k k k | a b c d | k k k k)`
    The input is extended by filling all
    values beyond the edge with the same constant value k = 0.
    - `"wrap"`: `(a b c d | a b c d | a b c d)`
    The input is extended by wrapping around to the opposite edge.
    - `"nearest"`: `(a a a a | a b c d | d d d d)`
    The input is extended by the nearest pixel.
    Input shape:
        4D tensor with shape: `(samples, height, width, channels)`,
            in `"channels_last"` format.
    Output shape:
        4D tensor with shape: `(samples, height, width, channels)`,
            in `"channels_last"` format.
    Returns:
        Image(s) with the same type and shape as `images`, with the given
        transform(s) applied. Transformed coordinates outside of the input image
        will be filled with zeros.
    """
    with backend.name_scope(name or "transform"):
        if output_shape is None:
            output_shape = tf.shape(images)[1:3]
            if not tf.executing_eagerly():
                output_shape_value = tf.get_static_value(output_shape)
                if output_shape_value is not None:
                    output_shape = output_shape_value
        output_shape = tf.convert_to_tensor(
            output_shape, tf.int32, name="output_shape"
        )
        if not output_shape.get_shape().is_compatible_with([2]):
            raise ValueError(
                "output_shape must be a 1-D Tensor of 2 elements: "
                "new_height, new_width, instead got "
                f"output_shape={output_shape}"
            )
        fill_value = tf.convert_to_tensor(
            fill_value, tf.float32, name="fill_value"
        )
        return tf.raw_ops.ImageProjectiveTransformV3(
            images=images,
            output_shape=output_shape,
            fill_value=fill_value,
            transforms=transforms,
            fill_mode=fill_mode.upper(),
            interpolation=interpolation.upper(),
        )
def get_rotation_matrix(angles, image_height, image_width, name=None):
    """Returns projective transform(s) for the given angle(s).
    Args:
        angles: A scalar angle to rotate all images by,
            or (for batches of images) a vector with an angle to
            rotate each image in the batch. The rank must be
            statically known (the shape is not `TensorShape(None)`).
        image_height: Height of the image(s) to be transformed.
        image_width: Width of the image(s) to be transformed.
        name: The name of the op.
    Returns:
        A tensor of shape (num_images, 8).
            Projective transforms which can be given
            to operation `image_projective_transform_v2`.
            If one row of transforms is
            [a0, a1, a2, b0, b1, b2, c0, c1], then it maps the *output* point
            `(x, y)` to a transformed *input* point
            `(x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k)`,
            where `k = c0 x + c1 y + 1`.
    """
    with backend.name_scope(name or "rotation_matrix"):
        x_offset = (
            (image_width - 1)
            - (
                tf.cos(angles) * (image_width - 1)
                - tf.sin(angles) * (image_height - 1)
            )
        ) / 2.0
        y_offset = (
            (image_height - 1)
            - (
                tf.sin(angles) * (image_width - 1)
                + tf.cos(angles) * (image_height - 1)
            )
        ) / 2.0
        num_angles = tf.shape(angles)[0]
        return tf.concat(
            values=[
                tf.cos(angles)[:, None],
                -tf.sin(angles)[:, None],
                x_offset[:, None],
                tf.sin(angles)[:, None],
                tf.cos(angles)[:, None],
                y_offset[:, None],
                tf.zeros((num_angles, 2), tf.float32),
            ],
            axis=1,
        )
