@keras_export("keras.backend.sign")
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def sign(x):
    """Element-wise sign.
    Args:
        x: Tensor or variable.
    Returns:
        A tensor.
    """
    return tf.sign(x)
