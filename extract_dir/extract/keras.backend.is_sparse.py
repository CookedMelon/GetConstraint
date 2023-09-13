@keras_export("keras.backend.is_sparse")
@doc_controls.do_not_generate_docs
def is_sparse(tensor):
    """Returns whether a tensor is a sparse tensor.
    Args:
        tensor: A tensor instance.
    Returns:
        A boolean.
    Example:
    >>> a = tf.keras.backend.placeholder((2, 2), sparse=False)
    >>> print(tf.keras.backend.is_sparse(a))
    False
    >>> b = tf.keras.backend.placeholder((2, 2), sparse=True)
    >>> print(tf.keras.backend.is_sparse(b))
    True
    """
    spec = getattr(tensor, "_type_spec", None)
    if spec is not None:
        return isinstance(spec, tf.SparseTensorSpec)
    return isinstance(tensor, tf.SparseTensor)
