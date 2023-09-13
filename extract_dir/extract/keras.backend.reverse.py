@keras_export("keras.backend.reverse")
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def reverse(x, axes):
    """Reverse a tensor along the specified axes.
    Args:
        x: Tensor to reverse.
        axes: Integer or iterable of integers.
            Axes to reverse.
    Returns:
        A tensor.
    """
    if isinstance(axes, int):
        axes = [axes]
    return tf.reverse(x, axes)
