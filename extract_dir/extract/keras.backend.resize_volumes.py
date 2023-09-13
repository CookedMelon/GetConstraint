@keras_export("keras.backend.resize_volumes")
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def resize_volumes(x, depth_factor, height_factor, width_factor, data_format):
    """Resizes the volume contained in a 5D tensor.
    Args:
        x: Tensor or variable to resize.
        depth_factor: Positive integer.
        height_factor: Positive integer.
        width_factor: Positive integer.
        data_format: One of `"channels_first"`, `"channels_last"`.
    Returns:
        A tensor.
    Raises:
        ValueError: if `data_format` is neither
            `channels_last` or `channels_first`.
    """
    if data_format == "channels_first":
        output = repeat_elements(x, depth_factor, axis=2)
        output = repeat_elements(output, height_factor, axis=3)
        output = repeat_elements(output, width_factor, axis=4)
        return output
    elif data_format == "channels_last":
        output = repeat_elements(x, depth_factor, axis=1)
        output = repeat_elements(output, height_factor, axis=2)
        output = repeat_elements(output, width_factor, axis=3)
        return output
    else:
        raise ValueError("Invalid data_format: " + str(data_format))
