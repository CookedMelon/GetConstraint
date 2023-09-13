@keras_export("keras.backend.batch_flatten")
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def batch_flatten(x):
    """Turn a nD tensor into a 2D tensor with same 0th dimension.
    In other words, it flattens each data samples of a batch.
    Args:
        x: A tensor or variable.
    Returns:
        A tensor.
    Examples:
      Flattening a 3D tensor to 2D by collapsing the last dimension.
    >>> x_batch = tf.keras.backend.ones(shape=(2, 3, 4, 5))
    >>> x_batch_flatten = batch_flatten(x_batch)
    >>> tf.keras.backend.int_shape(x_batch_flatten)
    (2, 60)
    """
    x = tf.reshape(x, tf.stack([-1, prod(shape(x)[1:])]))
    return x
