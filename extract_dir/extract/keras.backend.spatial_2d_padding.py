@keras_export("keras.backend.spatial_2d_padding")
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def spatial_2d_padding(x, padding=((1, 1), (1, 1)), data_format=None):
    """Pads the 2nd and 3rd dimensions of a 4D tensor.
    Args:
        x: Tensor or variable.
        padding: Tuple of 2 tuples, padding pattern.
        data_format: One of `channels_last` or `channels_first`.
    Returns:
        A padded 4D tensor.
    Raises:
        ValueError: if `data_format` is neither
            `channels_last` or `channels_first`.
    """
    assert len(padding) == 2
    assert len(padding[0]) == 2
    assert len(padding[1]) == 2
    if data_format is None:
        data_format = image_data_format()
    if data_format not in {"channels_first", "channels_last"}:
        raise ValueError("Unknown data_format: " + str(data_format))
    if data_format == "channels_first":
        pattern = [[0, 0], [0, 0], list(padding[0]), list(padding[1])]
    else:
        pattern = [[0, 0], list(padding[0]), list(padding[1]), [0, 0]]
    return tf.compat.v1.pad(x, pattern)
