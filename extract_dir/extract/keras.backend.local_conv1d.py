@keras_export("keras.backend.local_conv1d")
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def local_conv1d(inputs, kernel, kernel_size, strides, data_format=None):
    """Apply 1D conv with un-shared weights.
    Args:
        inputs: 3D tensor with shape:
            (batch_size, steps, input_dim)
            if data_format is "channels_last" or
            (batch_size, input_dim, steps)
            if data_format is "channels_first".
        kernel: the unshared weight for convolution,
            with shape (output_length, feature_dim, filters).
        kernel_size: a tuple of a single integer,
            specifying the length of the 1D convolution window.
        strides: a tuple of a single integer,
            specifying the stride length of the convolution.
        data_format: the data format, channels_first or channels_last.
    Returns:
        A 3d tensor with shape:
        (batch_size, output_length, filters)
        if data_format='channels_first'
        or 3D tensor with shape:
        (batch_size, filters, output_length)
        if data_format='channels_last'.
    """
    output_shape = (kernel.shape[0],)
    return local_conv(
        inputs, kernel, kernel_size, strides, output_shape, data_format
    )
