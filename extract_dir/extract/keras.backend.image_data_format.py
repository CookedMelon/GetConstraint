@keras_export("keras.backend.image_data_format")
@tf.__internal__.dispatch.add_dispatch_support
def image_data_format():
    """Returns the default image data format convention.
    Returns:
        A string, either `'channels_first'` or `'channels_last'`
    Example:
    >>> tf.keras.backend.image_data_format()
    'channels_last'
    """
    return _IMAGE_DATA_FORMAT
