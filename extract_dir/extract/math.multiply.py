@tf_export("math.multiply", "multiply")
@dispatch.register_binary_elementwise_api
@dispatch.add_dispatch_support
def multiply(x, y, name=None):
  """Returns an element-wise x * y.
  For example:
  >>> x = tf.constant(([1, 2, 3, 4]))
  >>> tf.math.multiply(x, x)
  <tf.Tensor: shape=(4,), dtype=..., numpy=array([ 1,  4,  9, 16], dtype=int32)>
  Since `tf.math.multiply` will convert its arguments to `Tensor`s, you can also
  pass in non-`Tensor` arguments:
  >>> tf.math.multiply(7,6)
  <tf.Tensor: shape=(), dtype=int32, numpy=42>
  If `x.shape` is not the same as `y.shape`, they will be broadcast to a
  compatible shape. (More about broadcasting
  [here](https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).)
  For example:
  >>> x = tf.ones([1, 2]);
  >>> y = tf.ones([2, 1]);
  >>> x * y  # Taking advantage of operator overriding
  <tf.Tensor: shape=(2, 2), dtype=float32, numpy=
  array([[1., 1.],
       [1., 1.]], dtype=float32)>
  The reduction version of this elementwise operation is `tf.math.reduce_prod`
  Args:
    x: A Tensor. Must be one of the following types: `bfloat16`,
      `half`, `float32`, `float64`, `uint8`, `int8`, `uint16`,
      `int16`, `int32`, `int64`, `complex64`, `complex128`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).
  Returns:
  A `Tensor`.  Has the same type as `x`.
  Raises:
   * InvalidArgumentError: When `x` and `y` have incompatible shapes or types.
  """
  return gen_math_ops.mul(x, y, name)
