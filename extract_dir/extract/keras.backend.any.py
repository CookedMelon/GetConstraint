@keras_export("keras.backend.any")
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def any(x, axis=None, keepdims=False):
    """Bitwise reduction (logical OR).
    Args:
        x: Tensor or variable.
        axis: axis along which to perform the reduction.
        keepdims: whether the drop or broadcast the reduction axes.
    Returns:
        A uint8 tensor (0s and 1s).
    """
    x = tf.cast(x, tf.bool)
    return tf.reduce_any(x, axis, keepdims)
