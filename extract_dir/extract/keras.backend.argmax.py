@keras_export("keras.backend.argmax")
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def argmax(x, axis=-1):
    """Returns the index of the maximum value along an axis.
    Args:
        x: Tensor or variable.
        axis: axis along which to perform the reduction.
    Returns:
        A tensor.
    """
    return tf.argmax(x, axis)
