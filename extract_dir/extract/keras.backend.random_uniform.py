@keras_export("keras.backend.random_uniform")
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def random_uniform(shape, minval=0.0, maxval=1.0, dtype=None, seed=None):
    """Returns a tensor with uniform distribution of values.
    Args:
        shape: A tuple of integers, the shape of tensor to create.
        minval: A float, lower boundary of the uniform distribution
            to draw samples.
        maxval: A float, upper boundary of the uniform distribution
            to draw samples.
        dtype: String, dtype of returned tensor.
        seed: Integer, random seed.
    Returns:
        A tensor.
    Example:
    >>> random_uniform_tensor = tf.keras.backend.random_uniform(shape=(2,3),
    ... minval=0.0, maxval=1.0)
    >>> random_uniform_tensor
    <tf.Tensor: shape=(2, 3), dtype=float32, numpy=...,
    dtype=float32)>
    """
    if dtype is None:
        dtype = floatx()
    if seed is None:
        seed = np.random.randint(10e6)
    return tf.random.uniform(
        shape, minval=minval, maxval=maxval, dtype=dtype, seed=seed
    )
