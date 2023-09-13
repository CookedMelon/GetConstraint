@keras_export("keras.utils.save_img", "keras.preprocessing.image.save_img")
def save_img(path, x, data_format=None, file_format=None, scale=True, **kwargs):
    """Saves an image stored as a Numpy array to a path or file object.
    Args:
        path: Path or file object.
        x: Numpy array.
        data_format: Image data format, either `"channels_first"` or
          `"channels_last"`.
        file_format: Optional file format override. If omitted, the format to
          use is determined from the filename extension. If a file object was
          used instead of a filename, this parameter should always be used.
        scale: Whether to rescale image values to be within `[0, 255]`.
        **kwargs: Additional keyword arguments passed to `PIL.Image.save()`.
    """
    if data_format is None:
        data_format = backend.image_data_format()
    img = array_to_img(x, data_format=data_format, scale=scale)
    if img.mode == "RGBA" and (file_format == "jpg" or file_format == "jpeg"):
        warnings.warn(
            "The JPG format does not support RGBA images, converting to RGB."
        )
        img = img.convert("RGB")
    img.save(path, format=file_format, **kwargs)
