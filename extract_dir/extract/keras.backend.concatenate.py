@keras_export("keras.backend.concatenate")
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def concatenate(tensors, axis=-1):
    """Concatenates a list of tensors alongside the specified axis.
    Args:
        tensors: list of tensors to concatenate.
        axis: concatenation axis.
    Returns:
        A tensor.
    Example:
        >>> a = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> b = tf.constant([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
        >>> tf.keras.backend.concatenate((a, b), axis=-1)
        <tf.Tensor: shape=(3, 6), dtype=int32, numpy=
        array([[ 1,  2,  3, 10, 20, 30],
               [ 4,  5,  6, 40, 50, 60],
               [ 7,  8,  9, 70, 80, 90]], dtype=int32)>
    """
    if axis < 0:
        rank = ndim(tensors[0])
        if rank:
            axis %= rank
        else:
            axis = 0
    if py_all(is_sparse(x) for x in tensors):
        return tf.compat.v1.sparse_concat(axis, tensors)
    elif py_all(isinstance(x, tf.RaggedTensor) for x in tensors):
        return tf.concat(tensors, axis)
    else:
        return tf.concat([to_dense(x) for x in tensors], axis)
