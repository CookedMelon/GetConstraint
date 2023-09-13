@keras_export("keras.backend.softmax")
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def softmax(x, axis=-1):
    """Softmax of a tensor.
    Args:
        x: A tensor or variable.
        axis: The dimension softmax would be performed on.
            The default is -1 which indicates the last dimension.
    Returns:
        A tensor.
    """
    return tf.nn.softmax(x, axis=axis)
