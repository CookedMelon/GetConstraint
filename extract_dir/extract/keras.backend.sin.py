@keras_export("keras.backend.sin")
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def sin(x):
    """Computes sin of x element-wise.
    Args:
        x: Tensor or variable.
    Returns:
        A tensor.
    """
    return tf.sin(x)
