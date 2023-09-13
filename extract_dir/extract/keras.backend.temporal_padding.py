@keras_export("keras.backend.temporal_padding")
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def temporal_padding(x, padding=(1, 1)):
    """Pads the middle dimension of a 3D tensor.
    Args:
        x: Tensor or variable.
        padding: Tuple of 2 integers, how many zeros to
            add at the start and end of dim 1.
    Returns:
        A padded 3D tensor.
    """
    assert len(padding) == 2
    pattern = [[0, 0], [padding[0], padding[1]], [0, 0]]
    return tf.compat.v1.pad(x, pattern)
