@keras_export("keras.backend.random_bernoulli")
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def random_bernoulli(shape, p=0.0, dtype=None, seed=None):
    """Returns a tensor with random bernoulli distribution of values.
    Args:
        shape: A tuple of integers, the shape of tensor to create.
        p: A float, `0. <= p <= 1`, probability of bernoulli distribution.
        dtype: String, dtype of returned tensor.
        seed: Integer, random seed.
    Returns:
        A tensor.
    """
    if dtype is None:
        dtype = floatx()
    if seed is None:
        seed = np.random.randint(10e6)
    return tf.where(
        tf.random.uniform(shape, dtype=dtype, seed=seed) <= p,
        tf.ones(shape, dtype=dtype),
        tf.zeros(shape, dtype=dtype),
    )
