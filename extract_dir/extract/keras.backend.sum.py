@keras_export("keras.backend.sum")
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def sum(x, axis=None, keepdims=False):
    """Sum of the values in a tensor, alongside the specified axis.
    Args:
        x: A tensor or variable.
        axis: An integer, the axis to sum over.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.
    Returns:
        A tensor with sum of `x`.
    """
    return tf.reduce_sum(x, axis, keepdims)
