@keras_export("keras.layers.minimum")
def minimum(inputs, **kwargs):
    """Functional interface to the `Minimum` layer.
    Args:
        inputs: A list of input tensors.
        **kwargs: Standard layer keyword arguments.
    Returns:
        A tensor, the element-wise minimum of the inputs.
    """
    return Minimum(**kwargs)(inputs)
