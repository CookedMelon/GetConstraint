@tf_export("get_static_value")
def constant_value(tensor, partial=False):  # pylint: disable=invalid-name
  """Returns the constant value of the given tensor, if efficiently calculable.
  This function attempts to partially evaluate the given tensor, and
  returns its value as a numpy ndarray if this succeeds.
  Example usage:
  >>> a = tf.constant(10)
  >>> tf.get_static_value(a)
  10
  >>> b = tf.constant(20)
  >>> tf.get_static_value(tf.add(a, b))
  30
  >>> # `tf.Variable` is not supported.
  >>> c = tf.Variable(30)
  >>> print(tf.get_static_value(c))
  None
  Using `partial` option is most relevant when calling `get_static_value` inside
  a `tf.function`. Setting it to `True` will return the results but for the
  values that cannot be evaluated will be `None`. For example:
  ```python
  class Foo:
    def __init__(self):
      self.a = tf.Variable(1)
      self.b = tf.constant(2)
    @tf.function
    def bar(self, partial):
      packed = tf.raw_ops.Pack(values=[self.a, self.b])
      static_val = tf.get_static_value(packed, partial=partial)
      tf.print(static_val)
  f = Foo()
  f.bar(partial=True)  # `array([None, array(2, dtype=int32)], dtype=object)`
  f.bar(partial=False)  # `None`
  ```
  Compatibility(V1): If `constant_value(tensor)` returns a non-`None` result, it
  will no longer be possible to feed a different value for `tensor`. This allows
  the result of this function to influence the graph that is constructed, and
  permits static shape optimizations.
  Args:
    tensor: The Tensor to be evaluated.
    partial: If True, the returned numpy array is allowed to have partially
      evaluated values. Values that can't be evaluated will be None.
  Returns:
    A numpy ndarray containing the constant value of the given `tensor`,
    or None if it cannot be calculated.
  Raises:
    TypeError: if tensor is not an ops.Tensor.
  """
  if isinstance(tensor, core.Value):
    try:
      return tensor.numpy()
    except errors_impl.UnimplementedError:
      # Some EagerTensors may not implement .numpy/resolve, e.g. parallel
      # tensors with multiple components on different devices.
      return None
  if not is_tensor(tensor):
    return tensor
  if not isinstance(tensor, core.Symbol):
    return None
  ret = _ConstantValue(tensor, partial)
  if ret is not None:
    # The caller may now depend on the constant value of `tensor`, so we
    # conservatively prevent it from being fed.
    tensor.graph.prevent_feeding(tensor)
  return ret
