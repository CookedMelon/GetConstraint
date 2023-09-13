@keras_export("keras.backend.zeros_like")
@doc_controls.do_not_generate_docs
def zeros_like(x, dtype=None, name=None):
    """Instantiates an all-zeros variable of the same shape as another tensor.
    Args:
        x: Keras variable or Keras tensor.
        dtype: dtype of returned Keras variable.
               `None` uses the dtype of `x`.
        name: name for the variable to create.
    Returns:
        A Keras variable with the shape of `x` filled with zeros.
    Example:
    ```python
    kvar = tf.keras.backend.variable(np.random.random((2,3)))
    kvar_zeros = tf.keras.backend.zeros_like(kvar)
    K.eval(kvar_zeros)
    # array([[ 0.,  0.,  0.], [ 0.,  0.,  0.]], dtype=float32)
    ```
    """
    return tf.zeros_like(x, dtype=dtype, name=name)
