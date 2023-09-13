@keras_export("keras.activations.linear")
@tf.__internal__.dispatch.add_dispatch_support
def linear(x):
    """Linear activation function (pass-through).
    Example:
    >>> a = tf.constant([-3.0, -1.0, 0.0, 1.0, 3.0], dtype = tf.float32)
    >>> b = tf.keras.activations.linear(a)
    >>> b.numpy()
    array([-3., -1.,  0.,  1.,  3.], dtype=float32)
    Args:
        x: Input tensor.
    Returns:
        The input, unmodified.
    """
    return x
