@keras_export("keras.backend.minimum")
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def minimum(x, y):
    """Element-wise minimum of two tensors.
    Args:
        x: Tensor or variable.
        y: Tensor or variable.
    Returns:
        A tensor.
    """
    return tf.minimum(x, y)
