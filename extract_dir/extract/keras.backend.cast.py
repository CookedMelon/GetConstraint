@keras_export("keras.backend.cast")
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def cast(x, dtype):
    """Casts a tensor to a different dtype and returns it.
    You can cast a Keras variable but it still returns a Keras tensor.
    Args:
        x: Keras tensor (or variable).
        dtype: String, either (`'float16'`, `'float32'`, or `'float64'`).
    Returns:
        Keras tensor with dtype `dtype`.
    Examples:
        Cast a float32 variable to a float64 tensor
    >>> input = tf.keras.backend.ones(shape=(1,3))
    >>> print(input)
    <tf.Variable 'Variable:0' shape=(1, 3) dtype=float32,
    numpy=array([[1., 1., 1.]], dtype=float32)>
    >>> cast_input = tf.keras.backend.cast(input, dtype='float64')
    >>> print(cast_input)
    tf.Tensor([[1. 1. 1.]], shape=(1, 3), dtype=float64)
    """
    return tf.cast(x, dtype)
