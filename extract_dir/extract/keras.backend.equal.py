@keras_export("keras.backend.equal")
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def equal(x, y):
    """Element-wise equality between two tensors.
    Args:
        x: Tensor or variable.
        y: Tensor or variable.
    Returns:
        A bool tensor.
    """
    return tf.equal(x, y)
