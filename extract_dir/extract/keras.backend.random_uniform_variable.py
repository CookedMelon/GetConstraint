@keras_export("keras.backend.random_uniform_variable")
@doc_controls.do_not_generate_docs
def random_uniform_variable(shape, low, high, dtype=None, name=None, seed=None):
    """Instantiates a variable with values drawn from a uniform distribution.
    Args:
        shape: Tuple of integers, shape of returned Keras variable.
        low: Float, lower boundary of the output interval.
        high: Float, upper boundary of the output interval.
        dtype: String, dtype of returned Keras variable.
        name: String, name of returned Keras variable.
        seed: Integer, random seed.
    Returns:
        A Keras variable, filled with drawn samples.
    Example:
    >>> kvar = tf.keras.backend.random_uniform_variable(shape=(2,3),
    ... low=0.0, high=1.0)
    >>> kvar
    <tf.Variable 'Variable:0' shape=(2, 3) dtype=float32, numpy=...,
    dtype=float32)>
    """
    if dtype is None:
        dtype = floatx()
    tf_dtype = tf.as_dtype(dtype)
    if seed is None:
        # ensure that randomness is conditioned by the Numpy RNG
        seed = np.random.randint(10e8)
    value = tf.compat.v1.random_uniform_initializer(
        low, high, dtype=tf_dtype, seed=seed
    )(shape)
    return variable(value, dtype=dtype, name=name)
