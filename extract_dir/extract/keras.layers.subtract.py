@keras_export("keras.layers.subtract")
def subtract(inputs, **kwargs):
    """Functional interface to the `Subtract` layer.
    Args:
        inputs: A list of input tensors (exactly 2).
        **kwargs: Standard layer keyword arguments.
    Returns:
        A tensor, the difference of the inputs.
    Examples:
    ```python
        import keras
        input1 = keras.layers.Input(shape=(16,))
        x1 = keras.layers.Dense(8, activation='relu')(input1)
        input2 = keras.layers.Input(shape=(32,))
        x2 = keras.layers.Dense(8, activation='relu')(input2)
        subtracted = keras.layers.subtract([x1, x2])
        out = keras.layers.Dense(4)(subtracted)
        model = keras.models.Model(inputs=[input1, input2], outputs=out)
    ```
    """
    return Subtract(**kwargs)(inputs)
