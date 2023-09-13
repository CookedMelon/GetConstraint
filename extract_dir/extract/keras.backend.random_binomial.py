@keras_export("keras.backend.random_binomial")
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def random_binomial(shape, p=0.0, dtype=None, seed=None):
    """Returns a tensor with random binomial distribution of values.
    DEPRECATED, use `tf.keras.backend.random_bernoulli` instead.
    The binomial distribution with parameters `n` and `p` is the probability
    distribution of the number of successful Bernoulli process. Only supports
    `n` = 1 for now.
    Args:
        shape: A tuple of integers, the shape of tensor to create.
        p: A float, `0. <= p <= 1`, probability of binomial distribution.
        dtype: String, dtype of returned tensor.
        seed: Integer, random seed.
    Returns:
        A tensor.
    Example:
    >>> random_binomial_tensor = tf.keras.backend.random_binomial(shape=(2,3),
    ... p=0.5)
    >>> random_binomial_tensor
    <tf.Tensor: shape=(2, 3), dtype=float32, numpy=...,
    dtype=float32)>
    """
    warnings.warn(
        "`tf.keras.backend.random_binomial` is deprecated, "
        "and will be removed in a future version."
        "Please use `tf.keras.backend.random_bernoulli` instead.",
        stacklevel=2,
    )
    return random_bernoulli(shape, p, dtype, seed)
