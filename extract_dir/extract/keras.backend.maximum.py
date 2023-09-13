@keras_export("keras.backend.maximum")
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def maximum(x, y):
    """Element-wise maximum of two tensors.
    Args:
        x: Tensor or variable.
        y: Tensor or variable.
    Returns:
        A tensor with the element wise maximum value(s) of `x` and `y`.
    Examples:
    >>> x = tf.Variable([[1, 2], [3, 4]])
    >>> y = tf.Variable([[2, 1], [0, -1]])
    >>> m = tf.keras.backend.maximum(x, y)
    >>> m
    <tf.Tensor: shape=(2, 2), dtype=int32, numpy=
    array([[2, 2],
           [3, 4]], dtype=int32)>
    """
    return tf.maximum(x, y)
