@keras_export("keras.backend.expand_dims")
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def expand_dims(x, axis=-1):
    """Adds a 1-sized dimension at index "axis".
    Args:
        x: A tensor or variable.
        axis: Position where to add a new axis.
    Returns:
        A tensor with expanded dimensions.
    """
    return tf.expand_dims(x, axis)
