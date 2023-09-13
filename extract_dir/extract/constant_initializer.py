@tf_export("constant_initializer", v1=[])
class Constant(Initializer):
  """Initializer that generates tensors with constant values.
  Initializers allow you to pre-specify an initialization strategy, encoded in
  the Initializer object, without knowing the shape and dtype of the variable
  being initialized.
  `tf.constant_initializer` returns an object which when called returns a tensor
  populated with the `value` specified in the constructor. This `value` must be
  convertible to the requested `dtype`.
  The argument `value` can be a scalar constant value, or a list of
  values. Scalars broadcast to whichever shape is requested from the
  initializer.
  If `value` is a list, then the length of the list must be equal to the number
  of elements implied by the desired shape of the tensor. If the total number of
  elements in `value` is not equal to the number of elements required by the
  tensor shape, the initializer will raise a `TypeError`.
  Examples:
  >>> def make_variables(k, initializer):
  ...   return (tf.Variable(initializer(shape=[k], dtype=tf.float32)),
  ...           tf.Variable(initializer(shape=[k, k], dtype=tf.float32)))
  >>> v1, v2 = make_variables(3, tf.constant_initializer(2.))
  >>> v1
  <tf.Variable ... shape=(3,) ... numpy=array([2., 2., 2.], dtype=float32)>
  >>> v2
  <tf.Variable ... shape=(3, 3) ... numpy=
  array([[2., 2., 2.],
         [2., 2., 2.],
         [2., 2., 2.]], dtype=float32)>
  >>> make_variables(4, tf.random_uniform_initializer(minval=-1., maxval=1.))
  (<tf.Variable...shape=(4,) dtype=float32...>, <tf.Variable...shape=(4, 4) ...
  >>> value = [0, 1, 2, 3, 4, 5, 6, 7]
  >>> init = tf.constant_initializer(value)
  >>> # Fitting shape
  >>> tf.Variable(init(shape=[2, 4], dtype=tf.float32))
  <tf.Variable ...
  array([[0., 1., 2., 3.],
         [4., 5., 6., 7.]], dtype=float32)>
  >>> # Larger shape
  >>> tf.Variable(init(shape=[3, 4], dtype=tf.float32))
  Traceback (most recent call last):
  ...
  TypeError: ...value has 8 elements, shape is (3, 4) with 12 elements...
  >>> # Smaller shape
  >>> tf.Variable(init(shape=[2, 3], dtype=tf.float32))
  Traceback (most recent call last):
  ...
  TypeError: ...value has 8 elements, shape is (2, 3) with 6 elements...
  Args:
    value: A Python scalar, list or tuple of values, or a N-dimensional numpy
      array. All elements of the initialized variable will be set to the
      corresponding value in the `value` argument.
  Raises:
    TypeError: If the input `value` is not one of the expected types.
  """
  def __init__(self, value=0):
    if not (np.isscalar(value) or isinstance(value, (list, tuple, np.ndarray))):
      raise TypeError(
          f"Invalid type for initial value: {type(value).__name__}. Expected "
          "Python scalar, list or tuple of values, or numpy.ndarray.")
    self.value = value
  def __call__(self, shape, dtype=None, **kwargs):
    """Returns a tensor object initialized as specified by the initializer.
    Args:
      shape: Shape of the tensor.
      dtype: Optional dtype of the tensor. If not provided the dtype of the
        tensor created will be the type of the inital value.
      **kwargs: Additional keyword arguments.
    Raises:
      TypeError: If the initializer cannot create a tensor of the requested
       dtype.
    """
    self._validate_kwargs(kwargs, support_partition=False)
    if dtype is not None:
      dtype = dtypes.as_dtype(dtype)
    return constant_op.constant(self.value, dtype=dtype, shape=shape)
  def get_config(self):
    return {"value": self.value}
