@tf_export("random.poisson", v1=[])
@dispatch.add_dispatch_support
def random_poisson_v2(shape, lam, dtype=dtypes.float32, seed=None, name=None):
  """Draws `shape` samples from each of the given Poisson distribution(s).
  `lam` is the rate parameter describing the distribution(s).
  Example:
  ```python
  samples = tf.random.poisson([10], [0.5, 1.5])
  # samples has shape [10, 2], where each slice [:, 0] and [:, 1] represents
  # the samples drawn from each distribution
  samples = tf.random.poisson([7, 5], [12.2, 3.3])
  # samples has shape [7, 5, 2], where each slice [:, :, 0] and [:, :, 1]
  # represents the 7x5 samples drawn from each of the two distributions
  ```
  Args:
    shape: A 1-D integer Tensor or Python array. The shape of the output samples
      to be drawn per "rate"-parameterized distribution.
    lam: A Tensor or Python value or N-D array of type `dtype`.
      `lam` provides the rate parameter(s) describing the poisson
      distribution(s) to sample.
    dtype: The type of the output: `float16`, `float32`, `float64`, `int32` or
      `int64`.
    seed: A Python integer. Used to create a random seed for the distributions.
      See
      `tf.random.set_seed`
      for behavior.
    name: Optional name for the operation.
  Returns:
    samples: a `Tensor` of shape `tf.concat([shape, tf.shape(lam)], axis=0)`
      with values of type `dtype`.
  """
  with ops.name_scope(name, "random_poisson", [lam, shape]):
    shape = ops.convert_to_tensor(shape, name="shape", dtype=dtypes.int32)
    seed1, seed2 = random_seed.get_seed(seed)
    result = gen_random_ops.random_poisson_v2(
        shape, lam, dtype=dtype, seed=seed1, seed2=seed2)
    _maybe_set_static_shape_helper(result, shape, lam)
    return result
