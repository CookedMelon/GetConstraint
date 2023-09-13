@keras_export("keras.backend.spatial_3d_padding")
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def spatial_3d_padding(x, padding=((1, 1), (1, 1), (1, 1)), data_format=None):
    """Pads 5D tensor with zeros along the depth, height, width dimensions.
    Pads these dimensions with respectively
    "padding[0]", "padding[1]" and "padding[2]" zeros left and right.
    For 'channels_last' data_format,
    the 2nd, 3rd and 4th dimension will be padded.
    For 'channels_first' data_format,
    the 3rd, 4th and 5th dimension will be padded.
    Args:
        x: Tensor or variable.
        padding: Tuple of 3 tuples, padding pattern.
        data_format: One of `channels_last` or `channels_first`.
    Returns:
        A padded 5D tensor.
    Raises:
        ValueError: if `data_format` is neither
            `channels_last` or `channels_first`.
    """
    assert len(padding) == 3
    assert len(padding[0]) == 2
    assert len(padding[1]) == 2
    assert len(padding[2]) == 2
    if data_format is None:
        data_format = image_data_format()
    if data_format not in {"channels_first", "channels_last"}:
        raise ValueError("Unknown data_format: " + str(data_format))
    if data_format == "channels_first":
        pattern = [
            [0, 0],
            [0, 0],
            [padding[0][0], padding[0][1]],
            [padding[1][0], padding[1][1]],
            [padding[2][0], padding[2][1]],
        ]
    else:
        pattern = [
            [0, 0],
            [padding[0][0], padding[0][1]],
            [padding[1][0], padding[1][1]],
            [padding[2][0], padding[2][1]],
            [0, 0],
        ]
    return tf.compat.v1.pad(x, pattern)
