@keras_export("keras.backend.cumprod")
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def cumprod(x, axis=0):
    """Cumulative product of the values in a tensor alongside `axis`.
    Args:
        x: A tensor or variable.
        axis: An integer, the axis to compute the product.
    Returns:
        A tensor of the cumulative product of values of `x` along `axis`.
    """
    return tf.math.cumprod(x, axis=axis)
