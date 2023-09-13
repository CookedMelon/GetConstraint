@keras_export("keras.backend.flatten")
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def flatten(x):
    """Flatten a tensor.
    Args:
        x: A tensor or variable.
    Returns:
        A tensor, reshaped into 1-D
    Example:
        >>> b = tf.constant([[1, 2], [3, 4]])
        >>> b
        <tf.Tensor: shape=(2, 2), dtype=int32, numpy=
        array([[1, 2],
               [3, 4]], dtype=int32)>
        >>> tf.keras.backend.flatten(b)
        <tf.Tensor: shape=(4,), dtype=int32,
            numpy=array([1, 2, 3, 4], dtype=int32)>
    """
    return tf.reshape(x, [-1])
