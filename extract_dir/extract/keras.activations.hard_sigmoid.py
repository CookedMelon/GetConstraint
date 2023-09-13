@keras_export("keras.activations.hard_sigmoid")
@tf.__internal__.dispatch.add_dispatch_support
def hard_sigmoid(x):
    """Hard sigmoid activation function.
    A faster approximation of the sigmoid activation.
    Piecewise linear approximation of the sigmoid function.
    Ref: 'https://en.wikipedia.org/wiki/Hard_sigmoid'
    Example:
    >>> a = tf.constant([-3.0, -1.0, 0.0, 1.0, 3.0], dtype = tf.float32)
    >>> b = tf.keras.activations.hard_sigmoid(a)
    >>> b.numpy()
    array([0. , 0.3, 0.5, 0.7, 1. ], dtype=float32)
    Args:
        x: Input tensor.
    Returns:
      The hard sigmoid activation, defined as:
        - `if x < -2.5: return 0`
        - `if x > 2.5: return 1`
        - `if -2.5 <= x <= 2.5: return 0.2 * x + 0.5`
    """
    return backend.hard_sigmoid(x)
