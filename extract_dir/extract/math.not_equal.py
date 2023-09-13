@tf_export("math.not_equal", "not_equal")
@dispatch.register_binary_elementwise_api
@dispatch.add_dispatch_support
def not_equal(x, y, name=None):
  """Returns the truth value of (x != y) element-wise.
  Performs a [broadcast](
  https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html) with the
  arguments and then an element-wise inequality comparison, returning a Tensor
  of boolean values.
  For example:
  >>> x = tf.constant([2, 4])
  >>> y = tf.constant(2)
  >>> tf.math.not_equal(x, y)
  <tf.Tensor: shape=(2,), dtype=bool, numpy=array([False,  True])>
  >>> x = tf.constant([2, 4])
  >>> y = tf.constant([2, 4])
  >>> tf.math.not_equal(x, y)
  <tf.Tensor: shape=(2,), dtype=bool, numpy=array([False,  False])>
  Args:
    x: A `tf.Tensor`.
    y: A `tf.Tensor`.
    name: A name for the operation (optional).
  Returns:
    A `tf.Tensor` of type bool with the same size as that of x or y.
  Raises:
    `tf.errors.InvalidArgumentError`: If shapes of arguments are incompatible
  """
  return gen_math_ops.not_equal(x, y, name=name)
