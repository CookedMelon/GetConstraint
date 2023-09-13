@keras_export("keras.backend.reshape")
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def reshape(x, shape):
    """Reshapes a tensor to the specified shape.
    Args:
        x: Tensor or variable.
        shape: Target shape tuple.
    Returns:
        A tensor.
    Example:
      >>> a = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
      >>> a
      <tf.Tensor: shape=(4, 3), dtype=int32, numpy=
      array([[ 1,  2,  3],
             [ 4,  5,  6],
             [ 7,  8,  9],
             [10, 11, 12]], dtype=int32)>
      >>> tf.keras.backend.reshape(a, shape=(2, 6))
      <tf.Tensor: shape=(2, 6), dtype=int32, numpy=
      array([[ 1,  2,  3,  4,  5,  6],
             [ 7,  8,  9, 10, 11, 12]], dtype=int32)>
    """
    return tf.reshape(x, shape)
