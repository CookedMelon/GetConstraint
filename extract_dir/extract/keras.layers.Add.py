@keras_export("keras.layers.Add")
class Add(_Merge):
    """Layer that adds a list of inputs.
    It takes as input a list of tensors,
    all of the same shape, and returns
    a single tensor (also of the same shape).
    Examples:
    >>> input_shape = (2, 3, 4)
    >>> x1 = tf.random.normal(input_shape)
    >>> x2 = tf.random.normal(input_shape)
    >>> y = tf.keras.layers.Add()([x1, x2])
    >>> print(y.shape)
    (2, 3, 4)
    Used in a functional model:
    >>> input1 = tf.keras.layers.Input(shape=(16,))
    >>> x1 = tf.keras.layers.Dense(8, activation='relu')(input1)
    >>> input2 = tf.keras.layers.Input(shape=(32,))
    >>> x2 = tf.keras.layers.Dense(8, activation='relu')(input2)
    >>> # equivalent to `added = tf.keras.layers.add([x1, x2])`
    >>> added = tf.keras.layers.Add()([x1, x2])
    >>> out = tf.keras.layers.Dense(4)(added)
    >>> model = tf.keras.models.Model(inputs=[input1, input2], outputs=out)
    """
    def _merge_function(self, inputs):
        output = inputs[0]
        for i in range(1, len(inputs)):
            output += inputs[i]
        return output
