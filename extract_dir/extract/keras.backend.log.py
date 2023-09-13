@keras_export("keras.backend.log")
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def log(x):
    """Element-wise log.
    Args:
        x: Tensor or variable.
    Returns:
        A tensor.
    """
    return tf.math.log(x)
