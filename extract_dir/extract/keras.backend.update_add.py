@keras_export("keras.backend.update_add")
@doc_controls.do_not_generate_docs
def update_add(x, increment):
    """Update the value of `x` by adding `increment`.
    Args:
        x: A Variable.
        increment: A tensor of same shape as `x`.
    Returns:
        The variable `x` updated.
    """
    return tf.compat.v1.assign_add(x, increment)
