@keras_export("keras.backend.square")
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def square(x):
    """Element-wise square.
    Args:
        x: Tensor or variable.
    Returns:
        A tensor.
    """
    return tf.square(x)
