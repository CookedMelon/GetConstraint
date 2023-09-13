@keras_export("keras.layers.multiply")
def multiply(inputs, **kwargs):
    """Functional interface to the `Multiply` layer.
    Example:
    >>> x1 = np.arange(3.0)
    >>> x2 = np.arange(3.0)
    >>> tf.keras.layers.multiply([x1, x2])
    <tf.Tensor: shape=(3,), dtype=float32, numpy=array([0., 1., 4.], ...)>
    Usage in a functional model:
    >>> input1 = tf.keras.layers.Input(shape=(16,))
    >>> x1 = tf.keras.layers.Dense(
    ...     8, activation='relu')(input1) #shape=(None, 8)
    >>> input2 = tf.keras.layers.Input(shape=(32,))
    >>> x2 = tf.keras.layers.Dense(
    ...     8, activation='relu')(input2) #shape=(None, 8)
    >>> out = tf.keras.layers.multiply([x1,x2]) #shape=(None, 8)
    >>> out = tf.keras.layers.Dense(4)(out)
    >>> model = tf.keras.models.Model(inputs=[input1, input2], outputs=out)
    Args:
        inputs: A list of input tensors.
        **kwargs: Standard layer keyword arguments.
    Returns:
        A tensor, the element-wise product of the inputs.
    """
    return Multiply(**kwargs)(inputs)
