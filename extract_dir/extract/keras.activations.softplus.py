@keras_export("keras.activations.softplus")
@tf.__internal__.dispatch.add_dispatch_support
def softplus(x):
    """Softplus activation function, `softplus(x) = log(exp(x) + 1)`.
    Example Usage:
    >>> a = tf.constant([-20, -1.0, 0.0, 1.0, 20], dtype = tf.float32)
    >>> b = tf.keras.activations.softplus(a)
    >>> b.numpy()
    array([2.0611537e-09, 3.1326166e-01, 6.9314718e-01, 1.3132616e+00,
             2.0000000e+01], dtype=float32)
    Args:
        x: Input tensor.
    Returns:
        The softplus activation: `log(exp(x) + 1)`.
    """
    return tf.math.softplus(x)
