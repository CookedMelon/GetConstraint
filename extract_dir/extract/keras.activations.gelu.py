@keras_export("keras.activations.gelu", v1=[])
@tf.__internal__.dispatch.add_dispatch_support
def gelu(x, approximate=False):
    """Applies the Gaussian error linear unit (GELU) activation function.
    Gaussian error linear unit (GELU) computes
    `x * P(X <= x)`, where `P(X) ~ N(0, 1)`.
    The (GELU) nonlinearity weights inputs by their value, rather than gates
    inputs by their sign as in ReLU.
    Example:
    >>> x = tf.constant([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=tf.float32)
    >>> y = tf.keras.activations.gelu(x)
    >>> y.numpy()
    array([-0.00404951, -0.15865529,  0.        ,  0.8413447 ,  2.9959507 ],
        dtype=float32)
    >>> y = tf.keras.activations.gelu(x, approximate=True)
    >>> y.numpy()
    array([-0.00363752, -0.15880796,  0.        ,  0.841192  ,  2.9963627 ],
        dtype=float32)
    Args:
        x: Input tensor.
        approximate: A `bool`, whether to enable approximation.
    Returns:
        The gaussian error linear activation:
        `0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))`
        if `approximate` is `True` or
        `x * P(X <= x) = 0.5 * x * (1 + erf(x / sqrt(2)))`,
        where `P(X) ~ N(0, 1)`,
        if `approximate` is `False`.
    Reference:
      - [Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415)
    """
    return tf.nn.gelu(x, approximate)
