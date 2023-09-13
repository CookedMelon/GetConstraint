@keras_export("keras.preprocessing.image.random_brightness")
def random_brightness(x, brightness_range, scale=True):
    """Performs a random brightness shift.
    Deprecated: `tf.keras.preprocessing.image.random_brightness` does not
    operate on tensors and is not recommended for new code. Prefer
    `tf.keras.layers.RandomBrightness` which provides equivalent functionality
    as a preprocessing layer. For more information, see the tutorial for
    [augmenting images](
    https://www.tensorflow.org/tutorials/images/data_augmentation), as well as
    the [preprocessing layer guide](
    https://www.tensorflow.org/guide/keras/preprocessing_layers).
    Args:
        x: Input tensor. Must be 3D.
        brightness_range: Tuple of floats; brightness range.
        scale: Whether to rescale the image such that minimum and maximum values
            are 0 and 255 respectively. Default: True.
    Returns:
        Numpy image tensor.
    Raises:
        ValueError if `brightness_range` isn't a tuple.
    """
    if len(brightness_range) != 2:
        raise ValueError(
            "`brightness_range should be tuple or list of two floats. "
            "Received: %s" % (brightness_range,)
        )
    u = np.random.uniform(brightness_range[0], brightness_range[1])
    return apply_brightness_shift(x, u, scale)
