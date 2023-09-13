@tf_export("random_uniform_initializer", v1=[])
class RandomUniform(Initializer):
  """Initializer that generates tensors with a uniform distribution.
  Initializers allow you to pre-specify an initialization strategy, encoded in
  the Initializer object, without knowing the shape and dtype of the variable
  being initialized.
  Examples:
  >>> def make_variables(k, initializer):
  ...   return (tf.Variable(initializer(shape=[k], dtype=tf.float32)),
  ...           tf.Variable(initializer(shape=[k, k], dtype=tf.float32)))
  >>> v1, v2 = make_variables(3, tf.ones_initializer())
  >>> v1
  <tf.Variable ... shape=(3,) ... numpy=array([1., 1., 1.], dtype=float32)>
  >>> v2
  <tf.Variable ... shape=(3, 3) ... numpy=
  array([[1., 1., 1.],
         [1., 1., 1.],
         [1., 1., 1.]], dtype=float32)>
  >>> make_variables(4, tf.random_uniform_initializer(minval=-1., maxval=1.))
  (<tf.Variable...shape=(4,) dtype=float32...>, <tf.Variable...shape=(4, 4) ...
  Args:
    minval: A python scalar or a scalar tensor. Lower bound of the range of
      random values to generate (inclusive).
    maxval: A python scalar or a scalar tensor. Upper bound of the range of
      random values to generate (exclusive).
    seed: A Python integer. Used to create random seeds. See
      `tf.random.set_seed` for behavior.
  """
  def __init__(self, minval=-0.05, maxval=0.05, seed=None):
    self.minval = minval
    self.maxval = maxval
    self.seed = seed
    self._random_generator = _RandomGenerator(seed)
  def __call__(self, shape, dtype=dtypes.float32, **kwargs):
    """Returns a tensor object initialized as specified by the initializer.
    Args:
      shape: Shape of the tensor.
      dtype: Optional dtype of the tensor. Only floating point and integer
        types are supported.
      **kwargs: Additional keyword arguments.
    Raises:
      ValueError: If the dtype is not numeric.
    """
    self._validate_kwargs(kwargs)
    dtype = dtypes.as_dtype(dtype)
    if not dtype.is_floating and not dtype.is_integer:
      raise ValueError("Argument `dtype` expected to be numeric or boolean. "
                       f"Received {dtype}.")
    if _PARTITION_SHAPE in kwargs:
      shape = kwargs[_PARTITION_SHAPE]
    return self._random_generator.random_uniform(shape, self.minval,
                                                 self.maxval, dtype)
  def get_config(self):
    return {
        "minval": self.minval,
        "maxval": self.maxval,
        "seed": self.seed
    }
