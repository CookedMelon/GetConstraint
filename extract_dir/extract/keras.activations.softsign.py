@keras_export("keras.activations.softsign")
@tf.__internal__.dispatch.add_dispatch_support
def softsign(x):
    """Softsign activation function, `softsign(x) = x / (abs(x) + 1)`.
    Example Usage:
    >>> a = tf.constant([-1.0, 0.0, 1.0], dtype = tf.float32)
    >>> b = tf.keras.activations.softsign(a)
    >>> b.numpy()
    array([-0.5,  0. ,  0.5], dtype=float32)
    Args:
        x: Input tensor.
    Returns:
        The softsign activation: `x / (abs(x) + 1)`.
    """
    return tf.math.softsign(x)
