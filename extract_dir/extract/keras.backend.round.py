@keras_export("keras.backend.round")
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def round(x):
    """Element-wise rounding to the closest integer.
    In case of tie, the rounding mode used is "half to even".
    Args:
        x: Tensor or variable.
    Returns:
        A tensor.
    """
    return tf.round(x)
