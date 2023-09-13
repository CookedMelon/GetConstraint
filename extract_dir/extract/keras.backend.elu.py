@keras_export("keras.backend.elu")
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def elu(x, alpha=1.0):
    """Exponential linear unit.
    Args:
        x: A tensor or variable to compute the activation function for.
        alpha: A scalar, slope of negative section.
    Returns:
        A tensor.
    """
    res = tf.nn.elu(x)
    if alpha == 1:
        return res
    else:
        return tf.where(x > 0, res, alpha * res)
