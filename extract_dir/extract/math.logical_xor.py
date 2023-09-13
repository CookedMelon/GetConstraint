@tf_export("math.logical_xor", v1=["math.logical_xor", "logical_xor"])
@dispatch.register_binary_elementwise_api
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints("logical_xor")
def logical_xor(x, y, name="LogicalXor"):
  """Logical XOR function.
  x ^ y = (x | y) & ~(x & y)
  Requires that `x` and `y` have the same shape or have
  [broadcast-compatible](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
  shapes. For example, `x` and `y` can be:
  - Two single elements of type `bool`
  - One `tf.Tensor` of type `bool` and one single `bool`, where the result will
    be calculated by applying logical XOR with the single element to each
    element in the larger Tensor.
  - Two `tf.Tensor` objects of type `bool` of the same shape. In this case,
    the result will be the element-wise logical XOR of the two input tensors.
  Usage:
  >>> a = tf.constant([True])
  >>> b = tf.constant([False])
  >>> tf.math.logical_xor(a, b)
  <tf.Tensor: shape=(1,), dtype=bool, numpy=array([ True])>
  >>> c = tf.constant([True])
  >>> x = tf.constant([False, True, True, False])
  >>> tf.math.logical_xor(c, x)
  <tf.Tensor: shape=(4,), dtype=bool, numpy=array([ True, False, False,  True])>
  >>> y = tf.constant([False, False, True, True])
  >>> z = tf.constant([False, True, False, True])
  >>> tf.math.logical_xor(y, z)
  <tf.Tensor: shape=(4,), dtype=bool, numpy=array([False,  True,  True, False])>
  Args:
      x: A `tf.Tensor` type bool.
      y: A `tf.Tensor` of type bool.
      name: A name for the operation (optional).
  Returns:
    A `tf.Tensor` of type bool with the same size as that of x or y.
  """
  # TODO(alemi) Make this a cwise op if people end up relying on it.
  return gen_math_ops.logical_and(
      gen_math_ops.logical_or(x, y),
      gen_math_ops.logical_not(gen_math_ops.logical_and(x, y)),
      name=name)
