@keras_export("keras.backend.gradients")
@doc_controls.do_not_generate_docs
def gradients(loss, variables):
    """Returns the gradients of `loss` w.r.t. `variables`.
    Args:
        loss: Scalar tensor to minimize.
        variables: List of variables.
    Returns:
        A gradients tensor.
    """
    return tf.compat.v1.gradients(
        loss, variables, colocate_gradients_with_ops=True
    )
