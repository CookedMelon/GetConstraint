@keras_export("keras.backend.pow")
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def pow(x, a):
    """Element-wise exponentiation.
    Args:
        x: Tensor or variable.
        a: Python integer.
    Returns:
        A tensor.
    """
    return tf.pow(x, a)
