@keras_export("keras.backend.tanh")
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def tanh(x):
    """Element-wise tanh.
    Args:
        x: A tensor or variable.
    Returns:
        A tensor.
    """
    return tf.tanh(x)
