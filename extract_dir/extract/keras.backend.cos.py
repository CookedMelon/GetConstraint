@keras_export("keras.backend.cos")
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def cos(x):
    """Computes cos of x element-wise.
    Args:
        x: Tensor or variable.
    Returns:
        A tensor.
    """
    return tf.cos(x)
