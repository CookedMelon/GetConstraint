@keras_export("keras.backend.truncated_normal")
@tf.__internal__.dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def truncated_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    """Returns a tensor with truncated random normal distribution of values.
    The generated values follow a normal distribution
    with specified mean and standard deviation,
    except that values whose magnitude is more than
    two standard deviations from the mean are dropped and re-picked.
    Args:
        shape: A tuple of integers, the shape of tensor to create.
        mean: Mean of the values.
        stddev: Standard deviation of the values.
        dtype: String, dtype of returned tensor.
        seed: Integer, random seed.
    Returns:
        A tensor.
    """
    if dtype is None:
        dtype = floatx()
    if seed is None:
        seed = np.random.randint(10e6)
    return tf.random.truncated_normal(
        shape, mean, stddev, dtype=dtype, seed=seed
    )
