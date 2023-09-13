@keras_export("keras.backend.softplus")
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def softplus(x):
    """Softplus of a tensor.
    Args:
        x: A tensor or variable.
    Returns:
        A tensor.
    """
    return tf.math.softplus(x)
