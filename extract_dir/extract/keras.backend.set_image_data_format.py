@keras_export("keras.backend.set_image_data_format")
def set_image_data_format(data_format):
    """Sets the value of the image data format convention.
    Args:
        data_format: string. `'channels_first'` or `'channels_last'`.
    Example:
    >>> tf.keras.backend.image_data_format()
    'channels_last'
    >>> tf.keras.backend.set_image_data_format('channels_first')
    >>> tf.keras.backend.image_data_format()
    'channels_first'
    >>> tf.keras.backend.set_image_data_format('channels_last')
    Raises:
        ValueError: In case of invalid `data_format` value.
    """
    global _IMAGE_DATA_FORMAT
    accepted_formats = {"channels_last", "channels_first"}
    if data_format not in accepted_formats:
        raise ValueError(
            f"Unknown `data_format`: {data_format}. "
            f"Expected one of {accepted_formats}"
        )
    _IMAGE_DATA_FORMAT = str(data_format)
