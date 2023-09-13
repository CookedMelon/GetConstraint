@keras_export("keras.backend.dtype")
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def dtype(x):
    """Returns the dtype of a Keras tensor or variable, as a string.
    Args:
        x: Tensor or variable.
    Returns:
        String, dtype of `x`.
    Examples:
    >>> tf.keras.backend.dtype(tf.keras.backend.placeholder(shape=(2,4,5)))
    'float32'
    >>> tf.keras.backend.dtype(tf.keras.backend.placeholder(shape=(2,4,5),
    ...                                                     dtype='float32'))
    'float32'
    >>> tf.keras.backend.dtype(tf.keras.backend.placeholder(shape=(2,4,5),
    ...                                                     dtype='float64'))
    'float64'
    >>> kvar = tf.keras.backend.variable(np.array([[1, 2], [3, 4]]))
    >>> tf.keras.backend.dtype(kvar)
    'float32'
    >>> kvar = tf.keras.backend.variable(np.array([[1, 2], [3, 4]]),
    ...                                  dtype='float32')
    >>> tf.keras.backend.dtype(kvar)
    'float32'
    """
    return x.dtype.base_dtype.name
