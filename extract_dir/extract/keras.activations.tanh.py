@keras_export("keras.activations.tanh")
@tf.__internal__.dispatch.add_dispatch_support
def tanh(x):
    """Hyperbolic tangent activation function.
    Example:
    >>> a = tf.constant([-3.0, -1.0, 0.0, 1.0, 3.0], dtype = tf.float32)
    >>> b = tf.keras.activations.tanh(a)
    >>> b.numpy()
    array([-0.9950547, -0.7615942,  0.,  0.7615942,  0.9950547], dtype=float32)
    Args:
        x: Input tensor.
    Returns:
        Tensor of same shape and dtype of input `x`, with tanh activation:
        `tanh(x) = sinh(x)/cosh(x) = ((exp(x) - exp(-x))/(exp(x) + exp(-x)))`.
    """
    return tf.tanh(x)
