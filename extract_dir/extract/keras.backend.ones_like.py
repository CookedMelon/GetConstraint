@keras_export("keras.backend.ones_like")
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def ones_like(x, dtype=None, name=None):
    """Instantiates an all-ones variable of the same shape as another tensor.
    Args:
        x: Keras variable or tensor.
        dtype: String, dtype of returned Keras variable.
             None uses the dtype of x.
        name: String, name for the variable to create.
    Returns:
        A Keras variable with the shape of x filled with ones.
    Example:
    >>> kvar = tf.keras.backend.variable(np.random.random((2,3)))
    >>> kvar_ones = tf.keras.backend.ones_like(kvar)
    >>> tf.keras.backend.eval(kvar_ones)
    array([[1.,  1.,  1.],
           [1.,  1.,  1.]], dtype=float32)
    """
    return tf.ones_like(x, dtype=dtype, name=name)
