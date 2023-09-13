@keras_export("keras.backend.cumsum")
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def cumsum(x, axis=0):
    """Cumulative sum of the values in a tensor, alongside the specified axis.
    Args:
        x: A tensor or variable.
        axis: An integer, the axis to compute the sum.
    Returns:
        A tensor of the cumulative sum of values of `x` along `axis`.
    """
    return tf.cumsum(x, axis=axis)
