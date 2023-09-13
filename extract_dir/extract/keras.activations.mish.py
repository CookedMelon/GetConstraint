@keras_export("keras.activations.mish")
@tf.__internal__.dispatch.add_dispatch_support
def mish(x):
    """Mish activation function.
    It is defined as:
    ```python
    def mish(x):
        return x * tanh(softplus(x))
    ```
    where `softplus` is defined as:
    ```python
    def softplus(x):
        return log(exp(x) + 1)
    ```
    Example:
    >>> a = tf.constant([-3.0, -1.0, 0.0, 1.0], dtype = tf.float32)
    >>> b = tf.keras.activations.mish(a)
    >>> b.numpy()
    array([-0.14564745, -0.30340144,  0.,  0.86509836], dtype=float32)
    Args:
        x: Input tensor.
    Returns:
        The mish activation.
    Reference:
        - [Mish: A Self Regularized Non-Monotonic
        Activation Function](https://arxiv.org/abs/1908.08681)
    """
    return x * tf.math.tanh(tf.math.softplus(x))
