@tf_export("random_normal_initializer", v1=[])
class RandomNormal(Initializer):
  """Initializer that generates tensors with a normal distribution.
  Initializers allow you to pre-specify an initialization strategy, encoded in
  the Initializer object, without knowing the shape and dtype of the variable
  being initialized.
  Examples:
  >>> def make_variables(k, initializer):
  ...   return (tf.Variable(initializer(shape=[k], dtype=tf.float32)),
  ...           tf.Variable(initializer(shape=[k, k], dtype=tf.float32)))
  >>> v1, v2 = make_variables(3,
  ...                         tf.random_normal_initializer(mean=1., stddev=2.))
  >>> v1
  <tf.Variable ... shape=(3,) ... numpy=array([...], dtype=float32)>
  >>> v2
  <tf.Variable ... shape=(3, 3) ... numpy=
  ...
  >>> make_variables(4, tf.random_uniform_initializer(minval=-1., maxval=1.))
  (<tf.Variable...shape=(4,) dtype=float32...>, <tf.Variable...shape=(4, 4) ...
  Args:
    mean: a python scalar or a scalar tensor. Mean of the random values to
      generate.
    stddev: a python scalar or a scalar tensor. Standard deviation of the random
      values to generate.
    seed: A Python integer. Used to create random seeds. See
      `tf.random.set_seed` for behavior.
  """
  def __init__(self, mean=0.0, stddev=0.05, seed=None):
    self.mean = mean
    self.stddev = stddev
    self.seed = seed
    self._random_generator = _RandomGenerator(seed)
  def __call__(self, shape, dtype=dtypes.float32, **kwargs):
    """Returns a tensor object initialized as specified by the initializer.
    Args:
      shape: Shape of the tensor.
      dtype: Optional dtype of the tensor. Only floating point types are
        supported.
      **kwargs: Additional keyword arguments.
    Raises:
      ValueError: If the dtype is not floating point
    """
    self._validate_kwargs(kwargs)
    dtype = _assert_float_dtype(dtype)
    if _PARTITION_SHAPE in kwargs:
      shape = kwargs[_PARTITION_SHAPE]
    return self._random_generator.random_normal(shape, self.mean, self.stddev,
                                                dtype)
  def get_config(self):
    return {
        "mean": self.mean,
        "stddev": self.stddev,
        "seed": self.seed
    }
