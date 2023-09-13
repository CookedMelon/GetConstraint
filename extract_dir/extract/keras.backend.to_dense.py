@keras_export("keras.backend.to_dense")
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def to_dense(tensor):
    """Converts a sparse tensor into a dense tensor and returns it.
    Args:
        tensor: A tensor instance (potentially sparse).
    Returns:
        A dense tensor.
    Examples:
    >>> b = tf.keras.backend.placeholder((2, 2), sparse=True)
    >>> print(tf.keras.backend.is_sparse(b))
    True
    >>> c = tf.keras.backend.to_dense(b)
    >>> print(tf.keras.backend.is_sparse(c))
    False
    """
    if is_sparse(tensor):
        return tf.sparse.to_dense(tensor)
    else:
        return tensor
