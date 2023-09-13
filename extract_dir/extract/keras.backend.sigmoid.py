@keras_export("keras.backend.sigmoid")
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def sigmoid(x):
    """Element-wise sigmoid.
    Args:
        x: A tensor or variable.
    Returns:
        A tensor.
    """
    return tf.math.sigmoid(x)
