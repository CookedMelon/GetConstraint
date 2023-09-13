@keras_export("keras.backend.repeat")
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def repeat(x, n):
    """Repeats a 2D tensor.
    if `x` has shape (samples, dim) and `n` is `2`,
    the output will have shape `(samples, 2, dim)`.
    Args:
        x: Tensor or variable.
        n: Python integer, number of times to repeat.
    Returns:
        A tensor.
    Example:
        >>> b = tf.constant([[1, 2], [3, 4]])
        >>> b
        <tf.Tensor: shape=(2, 2), dtype=int32, numpy=
        array([[1, 2],
               [3, 4]], dtype=int32)>
        >>> tf.keras.backend.repeat(b, n=2)
        <tf.Tensor: shape=(2, 2, 2), dtype=int32, numpy=
        array([[[1, 2],
                [1, 2]],
               [[3, 4],
                [3, 4]]], dtype=int32)>
    """
    assert ndim(x) == 2
    x = tf.expand_dims(x, 1)
    pattern = tf.stack([1, n, 1])
    return tf.tile(x, pattern)
