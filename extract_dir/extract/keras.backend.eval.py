@keras_export("keras.backend.eval")
@doc_controls.do_not_generate_docs
def eval(x):
    """Evaluates the value of a variable.
    Args:
        x: A variable.
    Returns:
        A Numpy array.
    Examples:
    >>> kvar = tf.keras.backend.variable(np.array([[1, 2], [3, 4]]),
    ...                                  dtype='float32')
    >>> tf.keras.backend.eval(kvar)
    array([[1.,  2.],
           [3.,  4.]], dtype=float32)
    """
    return get_value(to_dense(x))
