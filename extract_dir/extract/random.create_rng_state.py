@tf_export("random.create_rng_state", "random.experimental.create_rng_state")
def create_rng_state(seed, alg):
  """Creates a RNG state from an integer or a vector.
  Example:
  >>> tf.random.create_rng_state(
  ...     1234, "philox")
  <tf.Tensor: shape=(3,), dtype=int64, numpy=array([1234,    0,    0])>
  >>> tf.random.create_rng_state(
  ...     [12, 34], "threefry")
  <tf.Tensor: shape=(2,), dtype=int64, numpy=array([12, 34])>
  Args:
    seed: an integer or 1-D numpy array.
    alg: the RNG algorithm. Can be a string, an `Algorithm` or an integer.
  Returns:
    a 1-D numpy array whose size depends on the algorithm.
  """
  alg = stateless_random_ops.convert_alg_to_int(alg)
  return _make_state_from_seed(seed, alg)
