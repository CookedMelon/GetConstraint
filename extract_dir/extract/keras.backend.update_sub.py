@keras_export("keras.backend.update_sub")
@doc_controls.do_not_generate_docs
def update_sub(x, decrement):
    """Update the value of `x` by subtracting `decrement`.
    Args:
        x: A Variable.
        decrement: A tensor of same shape as `x`.
    Returns:
        The variable `x` updated.
    """
    return tf.compat.v1.assign_sub(x, decrement)
