"/home/cc/Workspace/tfconstraint/keras/utils/image_utils.py"
@keras_export(
    "keras.utils.array_to_img", "keras.preprocessing.image.array_to_img"
)
def array_to_img(x, data_format=None, scale=True, dtype=None):
    """Converts a 3D Numpy array to a PIL Image instance.
    Usage:
    ```python
    from PIL import Image
    img = np.random.random(size=(100, 100, 3))
    pil_img = tf.keras.utils.array_to_img(img)
    ```
    Args:
        x: Input data, in any form that can be converted to a Numpy array.
        data_format: Image data format, can be either `"channels_first"` or
          `"channels_last"`. Defaults to `None`, in which case the global
          setting `tf.keras.backend.image_data_format()` is used (unless you
          changed it, it defaults to `"channels_last"`).
        scale: Whether to rescale the image such that minimum and maximum values
          are 0 and 255 respectively. Defaults to `True`.
        dtype: Dtype to use. Default to `None`, in which case the global setting
          `tf.keras.backend.floatx()` is used (unless you changed it, it
          defaults to `"float32"`)
    Returns:
        A PIL Image instance.
    Raises:
        ImportError: if PIL is not available.
        ValueError: if invalid `x` or `data_format` is passed.
    """
    if data_format is None:
        data_format = backend.image_data_format()
    if dtype is None:
        dtype = backend.floatx()
    if pil_image is None:
        raise ImportError(
            "Could not import PIL.Image. "
            "The use of `array_to_img` requires PIL."
        )
    x = np.asarray(x, dtype=dtype)
    if x.ndim != 3:
        raise ValueError(
            "Expected image array to have rank 3 (single image). "
            f"Got array with shape: {x.shape}"
        )
    if data_format not in {"channels_first", "channels_last"}:
        raise ValueError(f"Invalid data_format: {data_format}")
    # Original Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but target PIL image has format (width, height, channel)
    if data_format == "channels_first":
        x = x.transpose(1, 2, 0)
    if scale:
        x = x - np.min(x)
        x_max = np.max(x)
        if x_max != 0:
            x /= x_max
        x *= 255
    if x.shape[2] == 4:
        # RGBA
        return pil_image.fromarray(x.astype("uint8"), "RGBA")
    elif x.shape[2] == 3:
        # RGB
        return pil_image.fromarray(x.astype("uint8"), "RGB")
    elif x.shape[2] == 1:
        # grayscale
        if np.max(x) > 255:
            # 32-bit signed integer grayscale image. PIL mode "I"
            return pil_image.fromarray(x[:, :, 0].astype("int32"), "I")
        return pil_image.fromarray(x[:, :, 0].astype("uint8"), "L")
    else:
        raise ValueError(f"Unsupported channel number: {x.shape[2]}")
