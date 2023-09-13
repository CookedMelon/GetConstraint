@keras_export("keras.backend.int_shape")
@doc_controls.do_not_generate_docs
def int_shape(x):
    """Returns shape of tensor/variable as a tuple of int/None entries.
    Args:
        x: Tensor or variable.
    Returns:
        A tuple of integers (or None entries).
    Examples:
    >>> input = tf.keras.backend.placeholder(shape=(2, 4, 5))
    >>> tf.keras.backend.int_shape(input)
    (2, 4, 5)
    >>> val = np.array([[1, 2], [3, 4]])
    >>> kvar = tf.keras.backend.variable(value=val)
    >>> tf.keras.backend.int_shape(kvar)
    (2, 2)
    """
    try:
        shape = x.shape
        if not isinstance(shape, tuple):
            shape = tuple(shape.as_list())
        return shape
    except ValueError:
        return None
