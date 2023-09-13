@keras_export("keras.layers.average")
def average(inputs, **kwargs):
    """Functional interface to the `tf.keras.layers.Average` layer.
    Example:
    >>> x1 = np.ones((2, 2))
    >>> x2 = np.zeros((2, 2))
    >>> y = tf.keras.layers.Average()([x1, x2])
    >>> y.numpy().tolist()
    [[0.5, 0.5], [0.5, 0.5]]
    Usage in a functional model:
    >>> input1 = tf.keras.layers.Input(shape=(16,))
    >>> x1 = tf.keras.layers.Dense(8, activation='relu')(input1)
    >>> input2 = tf.keras.layers.Input(shape=(32,))
    >>> x2 = tf.keras.layers.Dense(8, activation='relu')(input2)
    >>> avg = tf.keras.layers.Average()([x1, x2])
    >>> out = tf.keras.layers.Dense(4)(avg)
    >>> model = tf.keras.models.Model(inputs=[input1, input2], outputs=out)
    Args:
        inputs: A list of input tensors.
        **kwargs: Standard layer keyword arguments.
    Returns:
        A tensor, the average of the inputs.
    Raises:
      ValueError: If there is a shape mismatch between the inputs and the shapes
        cannot be broadcasted to match.
    """
    return Average(**kwargs)(inputs)
