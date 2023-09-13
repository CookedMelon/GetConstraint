@tf_export("zeros_initializer", v1=[])
class Zeros(Initializer):
  """Initializer that generates tensors initialized to 0.
  Initializers allow you to pre-specify an initialization strategy, encoded in
  the Initializer object, without knowing the shape and dtype of the variable
  being initialized.
  Examples:
  >>> def make_variables(k, initializer):
  ...   return (tf.Variable(initializer(shape=[k], dtype=tf.float32)),
  ...           tf.Variable(initializer(shape=[k, k], dtype=tf.float32)))
  >>> v1, v2 = make_variables(3, tf.zeros_initializer())
  >>> v1
  <tf.Variable ... shape=(3,) ... numpy=array([0., 0., 0.], dtype=float32)>
  >>> v2
  <tf.Variable ... shape=(3, 3) ... numpy=
  array([[0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.]], dtype=float32)>
  >>> make_variables(4, tf.random_uniform_initializer(minval=-1., maxval=1.))
  (<tf.Variable...shape=(4,) dtype=float32...>, <tf.Variable...shape=(4, 4) ...
  """
  def __call__(self, shape, dtype=dtypes.float32, **kwargs):
    """Returns a tensor object initialized as specified by the initializer.
    Args:
      shape: Shape of the tensor.
      dtype: Optional dtype of the tensor. Only numeric or boolean dtypes are
       supported.
      **kwargs: Additional keyword arguments.
    Raises:
      ValuesError: If the dtype is not numeric or boolean.
    """
    self._validate_kwargs(kwargs)
    dtype = dtypes.as_dtype(dtype)
    if not dtype.is_numpy_compatible or dtype == dtypes.string:
      raise ValueError("Argument `dtype` expected to be numeric or boolean. "
                       f"Received {dtype}.")
    if _PARTITION_SHAPE in kwargs:
      shape = kwargs[_PARTITION_SHAPE]
    return array_ops.zeros(shape, dtype)
