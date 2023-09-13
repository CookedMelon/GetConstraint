@keras_export("keras.backend.std")
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def std(x, axis=None, keepdims=False):
    """Standard deviation of a tensor, alongside the specified axis.
    It is an alias to `tf.math.reduce_std`.
    Args:
        x: A tensor or variable. It should have numerical dtypes. Boolean type
          inputs will be converted to float.
        axis: An integer, the axis to compute the standard deviation. If `None`
          (the default), reduces all dimensions. Must be in the range
          `[-rank(x), rank(x))`.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`, the reduced dimension is retained
            with length 1.
    Returns:
        A tensor with the standard deviation of elements of `x` with same dtype.
        Boolean type input will be converted to float.
    """
    if x.dtype.base_dtype == tf.bool:
        x = tf.cast(x, floatx())
    return tf.math.reduce_std(x, axis=axis, keepdims=keepdims)
