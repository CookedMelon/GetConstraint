@keras_export("keras.backend.squeeze")
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def squeeze(x, axis):
    """Removes a 1-dimension from the tensor at index "axis".
    Args:
        x: A tensor or variable.
        axis: Axis to drop.
    Returns:
        A tensor with the same data as `x` but reduced dimensions.
    """
    return tf.squeeze(x, [axis])
