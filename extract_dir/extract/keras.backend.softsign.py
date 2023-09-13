@keras_export("keras.backend.softsign")
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def softsign(x):
    """Softsign of a tensor.
    Args:
        x: A tensor or variable.
    Returns:
        A tensor.
    """
    return tf.math.softsign(x)
