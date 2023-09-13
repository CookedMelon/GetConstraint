@keras_export("keras.backend.max")
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def max(x, axis=None, keepdims=False):
    """Maximum value in a tensor.
    Args:
        x: A tensor or variable.
        axis: An integer, the axis to find maximum values.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.
    Returns:
        A tensor with maximum values of `x`.
    """
    return tf.reduce_max(x, axis, keepdims)
