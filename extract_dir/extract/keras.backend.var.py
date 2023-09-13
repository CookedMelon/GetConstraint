@keras_export("keras.backend.var")
@doc_controls.do_not_generate_docs
def var(x, axis=None, keepdims=False):
    """Variance of a tensor, alongside the specified axis.
    Args:
        x: A tensor or variable.
        axis: An integer, the axis to compute the variance.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.
    Returns:
        A tensor with the variance of elements of `x`.
    """
    if x.dtype.base_dtype == tf.bool:
        x = tf.cast(x, floatx())
    return tf.math.reduce_variance(x, axis=axis, keepdims=keepdims)
