@keras_export("keras.backend.abs")
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def abs(x):
    """Element-wise absolute value.
    Args:
        x: Tensor or variable.
    Returns:
        A tensor.
    """
    return tf.abs(x)
