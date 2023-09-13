@keras_export("keras.backend.dropout")
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def dropout(x, level, noise_shape=None, seed=None):
    """Sets entries in `x` to zero at random, while scaling the entire tensor.
    Args:
        x: tensor
        level: fraction of the entries in the tensor
            that will be set to 0.
        noise_shape: shape for randomly generated keep/drop flags,
            must be broadcastable to the shape of `x`
        seed: random seed to ensure determinism.
    Returns:
        A tensor.
    """
    if seed is None:
        seed = np.random.randint(10e6)
    return tf.nn.dropout(x, rate=level, noise_shape=noise_shape, seed=seed)
