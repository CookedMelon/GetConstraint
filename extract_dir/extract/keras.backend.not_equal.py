@keras_export("keras.backend.not_equal")
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def not_equal(x, y):
    """Element-wise inequality between two tensors.
    Args:
        x: Tensor or variable.
        y: Tensor or variable.
    Returns:
        A bool tensor.
    """
    return tf.not_equal(x, y)
