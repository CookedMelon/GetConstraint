@keras_export("keras.backend.greater")
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def greater(x, y):
    """Element-wise truth value of (x > y).
    Args:
        x: Tensor or variable.
        y: Tensor or variable.
    Returns:
        A bool tensor.
    """
    return tf.greater(x, y)
