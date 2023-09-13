@keras_export("keras.backend.stop_gradient")
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def stop_gradient(variables):
    """Returns `variables` but with zero gradient w.r.t. every other variable.
    Args:
        variables: Tensor or list of tensors to consider constant with respect
          to any other variable.
    Returns:
        A single tensor or a list of tensors (depending on the passed argument)
        that has no gradient with respect to any other variable.
    """
    if isinstance(variables, (list, tuple)):
        return map(tf.stop_gradient, variables)
    return tf.stop_gradient(variables)
