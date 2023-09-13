@keras_export("keras.backend.argmin")
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def argmin(x, axis=-1):
    """Returns the index of the minimum value along an axis.
    Args:
        x: Tensor or variable.
        axis: axis along which to perform the reduction.
    Returns:
        A tensor.
    """
    return tf.argmin(x, axis)
