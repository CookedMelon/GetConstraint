@keras_export("keras.activations.softmax")
@tf.__internal__.dispatch.add_dispatch_support
def softmax(x, axis=-1):
    """Softmax converts a vector of values to a probability distribution.
    The elements of the output vector are in range (0, 1) and sum to 1.
    Each vector is handled independently. The `axis` argument sets which axis
    of the input the function is applied along.
    Softmax is often used as the activation for the last
    layer of a classification network because the result could be interpreted as
    a probability distribution.
    The softmax of each vector x is computed as
    `exp(x) / tf.reduce_sum(exp(x))`.
    The input values in are the log-odds of the resulting probability.
    Args:
      x : Input tensor.
      axis: Integer, axis along which the softmax normalization is applied.
    Returns:
      Tensor, output of softmax transformation (all values are non-negative
        and sum to 1).
    Examples:
    **Example 1: standalone usage**
    >>> inputs = tf.random.normal(shape=(32, 10))
    >>> outputs = tf.keras.activations.softmax(inputs)
    >>> tf.reduce_sum(outputs[0, :])  # Each sample in the batch now sums to 1
    <tf.Tensor: shape=(), dtype=float32, numpy=1.0000001>
    **Example 2: usage in a `Dense` layer**
    >>> layer = tf.keras.layers.Dense(32,
    ...                               activation=tf.keras.activations.softmax)
    """
    if x.shape.rank <= 1:
        raise ValueError(
            f"Cannot apply softmax to a tensor that is 1D. Received input: {x}"
        )
    if isinstance(axis, int):
        output = tf.nn.softmax(x, axis=axis)
    else:
        # nn.softmax does not support tuple axis.
        numerator = tf.exp(x - tf.reduce_max(x, axis=axis, keepdims=True))
        denominator = tf.reduce_sum(numerator, axis=axis, keepdims=True)
        output = numerator / denominator
    # Cache the logits to use for crossentropy loss.
    output._keras_logits = x
    return output
