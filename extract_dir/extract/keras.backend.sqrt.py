@keras_export("keras.backend.sqrt")
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def sqrt(x):
    """Element-wise square root.
       This function clips negative tensor values to 0 before computing the
       square root.
    Args:
        x: Tensor or variable.
    Returns:
        A tensor.
    """
    zero = _constant_to_tensor(0.0, x.dtype.base_dtype)
    x = tf.maximum(x, zero)
    return tf.sqrt(x)
