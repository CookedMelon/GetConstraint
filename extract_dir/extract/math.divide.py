@tf_export("math.divide", "divide")
@dispatch.register_binary_elementwise_api
@dispatch.add_dispatch_support
def divide(x, y, name=None):
  """Computes Python style division of `x` by `y`.
  For example:
  >>> x = tf.constant([16, 12, 11])
  >>> y = tf.constant([4, 6, 2])
  >>> tf.divide(x,y)
  <tf.Tensor: shape=(3,), dtype=float64,
  numpy=array([4. , 2. , 5.5])>
  Args:
    x: A `Tensor`
    y: A `Tensor`
    name: A name for the operation (optional).
  Returns:
    A `Tensor` with same shape as input
  """
  if name is not None:
    # Cannot use tensors operator overload, because it has no way to track
    # override names. Use a dummy class to track the runtime division behavior
    return DivideDelegateWithName(x, name) / y
  else:
    # We do conversion here to make sure at least x is a tensor.
    if not tensor_util.is_tf_type(x):
      dtype = y.dtype.base_dtype if tensor_util.is_tf_type(y) else None
      x = ops.convert_to_tensor(x, dtype=dtype)
    return x / y
