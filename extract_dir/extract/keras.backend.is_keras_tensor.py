@keras_export("keras.backend.is_keras_tensor")
def is_keras_tensor(x):
    """Returns whether `x` is a Keras tensor.
    A "Keras tensor" is a tensor that was returned by a Keras layer,
    (`Layer` class) or by `Input`.
    Args:
        x: A candidate tensor.
    Returns:
        A boolean: Whether the argument is a Keras tensor.
    Raises:
        ValueError: In case `x` is not a symbolic tensor.
    Examples:
    >>> np_var = np.array([1, 2])
    >>> # A numpy array is not a symbolic tensor.
    >>> tf.keras.backend.is_keras_tensor(np_var)
    Traceback (most recent call last):
    ...
    ValueError: Unexpectedly found an instance of type
    `<class 'numpy.ndarray'>`.
    Expected a symbolic tensor instance.
    >>> keras_var = tf.keras.backend.variable(np_var)
    >>> # A variable created with the keras backend is not a Keras tensor.
    >>> tf.keras.backend.is_keras_tensor(keras_var)
    False
    >>> keras_placeholder = tf.keras.backend.placeholder(shape=(2, 4, 5))
    >>> # A placeholder is a Keras tensor.
    >>> tf.keras.backend.is_keras_tensor(keras_placeholder)
    True
    >>> keras_input = tf.keras.layers.Input([10])
    >>> # An Input is a Keras tensor.
    >>> tf.keras.backend.is_keras_tensor(keras_input)
    True
    >>> keras_layer_output = tf.keras.layers.Dense(10)(keras_input)
    >>> # Any Keras layer output is a Keras tensor.
    >>> tf.keras.backend.is_keras_tensor(keras_layer_output)
    True
    """
    if not isinstance(
        x,
        (
            tf.Tensor,
            tf.Variable,
            tf.SparseTensor,
            tf.RaggedTensor,
            keras_tensor.KerasTensor,
        ),
    ):
        raise ValueError(
            "Unexpectedly found an instance of type `"
            + str(type(x))
            + "`. Expected a symbolic tensor instance."
        )
    if tf.compat.v1.executing_eagerly_outside_functions():
        return isinstance(x, keras_tensor.KerasTensor)
    return hasattr(x, "_keras_history")
