@keras_export("keras.activations.exponential")
@tf.__internal__.dispatch.add_dispatch_support
def exponential(x):
    """Exponential activation function.
    Example:
    >>> a = tf.constant([-3.0, -1.0, 0.0, 1.0, 3.0], dtype = tf.float32)
    >>> b = tf.keras.activations.exponential(a)
    >>> b.numpy()
    array([0.04978707,  0.36787945,  1.,  2.7182817 , 20.085537], dtype=float32)
    Args:
        x: Input tensor.
    Returns:
        Tensor with exponential activation: `exp(x)`.
    """
    return tf.exp(x)
