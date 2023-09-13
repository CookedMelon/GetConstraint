@keras_export("keras.backend.count_params")
@doc_controls.do_not_generate_docs
def count_params(x):
    """Returns the static number of elements in a variable or tensor.
    Args:
        x: Variable or tensor.
    Returns:
        Integer, the number of scalars in `x`.
    Example:
    >>> kvar = tf.keras.backend.zeros((2,3))
    >>> tf.keras.backend.count_params(kvar)
    6
    >>> tf.keras.backend.eval(kvar)
    array([[0.,  0.,  0.],
           [0.,  0.,  0.]], dtype=float32)
    """
    return np.prod(x.shape.as_list())
