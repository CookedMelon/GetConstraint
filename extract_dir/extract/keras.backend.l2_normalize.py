@keras_export("keras.backend.l2_normalize")
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def l2_normalize(x, axis=None):
    """Normalizes a tensor wrt the L2 norm alongside the specified axis.
    Args:
        x: Tensor or variable.
        axis: axis along which to perform normalization.
    Returns:
        A tensor.
    """
    return tf.linalg.l2_normalize(x, axis=axis)
