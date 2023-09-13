@keras_export("keras.activations.swish")
@tf.__internal__.dispatch.add_dispatch_support
def swish(x):
    """Swish activation function, `swish(x) = x * sigmoid(x)`.
    Swish activation function which returns `x*sigmoid(x)`.
    It is a smooth, non-monotonic function that consistently matches
    or outperforms ReLU on deep networks, it is unbounded above and
    bounded below.
    Example Usage:
    >>> a = tf.constant([-20, -1.0, 0.0, 1.0, 20], dtype = tf.float32)
    >>> b = tf.keras.activations.swish(a)
    >>> b.numpy()
    array([-4.1223075e-08, -2.6894143e-01,  0.0000000e+00,  7.3105860e-01,
              2.0000000e+01], dtype=float32)
    Args:
        x: Input tensor.
    Returns:
        The swish activation applied to `x` (see reference paper for details).
    Reference:
      - [Ramachandran et al., 2017](https://arxiv.org/abs/1710.05941)
    """
    return tf.nn.silu(x)
