@keras_export("keras.layers.add")
def add(inputs, **kwargs):
    """Functional interface to the `tf.keras.layers.Add` layer.
    Args:
        inputs: A list of input tensors with the same shape.
        **kwargs: Standard layer keyword arguments.
    Returns:
        A tensor as the sum of the inputs. It has the same shape as the inputs.
    Examples:
    >>> input_shape = (2, 3, 4)
    >>> x1 = tf.random.normal(input_shape)
    >>> x2 = tf.random.normal(input_shape)
    >>> y = tf.keras.layers.add([x1, x2])
    >>> print(y.shape)
    (2, 3, 4)
    Used in a functional model:
    >>> input1 = tf.keras.layers.Input(shape=(16,))
    >>> x1 = tf.keras.layers.Dense(8, activation='relu')(input1)
    >>> input2 = tf.keras.layers.Input(shape=(32,))
    >>> x2 = tf.keras.layers.Dense(8, activation='relu')(input2)
    >>> added = tf.keras.layers.add([x1, x2])
    >>> out = tf.keras.layers.Dense(4)(added)
    >>> model = tf.keras.models.Model(inputs=[input1, input2], outputs=out)
    """
    return Add(**kwargs)(inputs)
