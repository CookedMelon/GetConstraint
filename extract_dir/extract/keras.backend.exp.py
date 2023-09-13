@keras_export("keras.backend.exp")
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def exp(x):
    """Element-wise exponential.
    Args:
        x: Tensor or variable.
    Returns:
        A tensor.
    """
    return tf.exp(x)
