@keras_export("keras.backend.ndim")
@doc_controls.do_not_generate_docs
def ndim(x):
    """Returns the number of axes in a tensor, as an integer.
    Args:
        x: Tensor or variable.
    Returns:
        Integer (scalar), number of axes.
    Examples:
    >>> input = tf.keras.backend.placeholder(shape=(2, 4, 5))
    >>> val = np.array([[1, 2], [3, 4]])
    >>> kvar = tf.keras.backend.variable(value=val)
    >>> tf.keras.backend.ndim(input)
    3
    >>> tf.keras.backend.ndim(kvar)
    2
    """
    return x.shape.rank
