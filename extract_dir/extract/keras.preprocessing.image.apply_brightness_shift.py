@keras_export("keras.preprocessing.image.apply_brightness_shift")
def apply_brightness_shift(x, brightness, scale=True):
    """Performs a brightness shift.
    Args:
        x: Input tensor. Must be 3D.
        brightness: Float. The new brightness value.
        scale: Whether to rescale the image such that minimum and maximum values
            are 0 and 255 respectively. Default: True.
    Returns:
        Numpy image tensor.
    Raises:
        ImportError: if PIL is not available.
    """
    if ImageEnhance is None:
        raise ImportError(
            "Using brightness shifts requires PIL. Install PIL or Pillow."
        )
    x_min, x_max = np.min(x), np.max(x)
    local_scale = (x_min < 0) or (x_max > 255)
    x = image_utils.array_to_img(x, scale=local_scale or scale)
    x = imgenhancer_Brightness = ImageEnhance.Brightness(x)
    x = imgenhancer_Brightness.enhance(brightness)
    x = image_utils.img_to_array(x)
    if not scale and local_scale:
        x = x / 255 * (x_max - x_min) + x_min
    return x
