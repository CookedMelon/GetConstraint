@keras_export("keras.activations.sigmoid")
@tf.__internal__.dispatch.add_dispatch_support
def sigmoid(x):
    """Sigmoid activation function, `sigmoid(x) = 1 / (1 + exp(-x))`.
    Applies the sigmoid activation function. For small values (<-5),
    `sigmoid` returns a value close to zero, and for large values (>5)
    the result of the function gets close to 1.
    Sigmoid is equivalent to a 2-element Softmax, where the second element is
    assumed to be zero. The sigmoid function always returns a value between
    0 and 1.
    Example:
    >>> a = tf.constant([-20, -1.0, 0.0, 1.0, 20], dtype = tf.float32)
    >>> b = tf.keras.activations.sigmoid(a)
    >>> b.numpy()
    array([2.0611537e-09, 2.6894143e-01, 5.0000000e-01, 7.3105860e-01,
             1.0000000e+00], dtype=float32)
    Args:
        x: Input tensor.
    Returns:
        Tensor with the sigmoid activation: `1 / (1 + exp(-x))`.
    """
    output = tf.sigmoid(x)
    # Cache the logits to use for crossentropy loss.
    output._keras_logits = x
    return output
