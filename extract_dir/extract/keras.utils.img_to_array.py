@keras_export(
    "keras.utils.img_to_array", "keras.preprocessing.image.img_to_array"
)
def img_to_array(img, data_format=None, dtype=None):
    """Converts a PIL Image instance to a Numpy array.
    Usage:
    ```python
    from PIL import Image
    img_data = np.random.random(size=(100, 100, 3))
    img = tf.keras.utils.array_to_img(img_data)
    array = tf.keras.utils.image.img_to_array(img)
    ```
    Args:
        img: Input PIL Image instance.
        data_format: Image data format, can be either `"channels_first"` or
          `"channels_last"`. Defaults to `None`, in which case the global
          setting `tf.keras.backend.image_data_format()` is used (unless you
          changed it, it defaults to `"channels_last"`).
        dtype: Dtype to use. Default to `None`, in which case the global setting
          `tf.keras.backend.floatx()` is used (unless you changed it, it
          defaults to `"float32"`).
    Returns:
        A 3D Numpy array.
    Raises:
        ValueError: if invalid `img` or `data_format` is passed.
    """
    if data_format is None:
        data_format = backend.image_data_format()
    if dtype is None:
        dtype = backend.floatx()
    if data_format not in {"channels_first", "channels_last"}:
        raise ValueError(f"Unknown data_format: {data_format}")
    # Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but original PIL image has format (width, height, channel)
    x = np.asarray(img, dtype=dtype)
    if len(x.shape) == 3:
        if data_format == "channels_first":
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if data_format == "channels_first":
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise ValueError(f"Unsupported image shape: {x.shape}")
    return x
