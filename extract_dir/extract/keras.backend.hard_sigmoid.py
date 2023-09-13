@keras_export("keras.backend.hard_sigmoid")
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def hard_sigmoid(x):
    """Segment-wise linear approximation of sigmoid.
    Faster than sigmoid.
    Returns `0.` if `x < -2.5`, `1.` if `x > 2.5`.
    In `-2.5 <= x <= 2.5`, returns `0.2 * x + 0.5`.
    Args:
        x: A tensor or variable.
    Returns:
        A tensor.
    """
    point_two = _constant_to_tensor(0.2, x.dtype.base_dtype)
    point_five = _constant_to_tensor(0.5, x.dtype.base_dtype)
    x = tf.multiply(x, point_two)
    x = tf.add(x, point_five)
    x = tf.clip_by_value(x, 0.0, 1.0)
    return x
