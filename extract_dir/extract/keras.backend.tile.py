@keras_export("keras.backend.tile")
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def tile(x, n):
    """Creates a tensor by tiling `x` by `n`.
    Args:
        x: A tensor or variable
        n: A list of integer. The length must be the same as the number of
            dimensions in `x`.
    Returns:
        A tiled tensor.
    """
    if isinstance(n, int):
        n = [n]
    return tf.tile(x, n)
