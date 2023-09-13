@tf_export("identity")
@dispatch.add_dispatch_support
def identity(input, name=None):  # pylint: disable=redefined-builtin
  r"""Return a Tensor with the same shape and contents as input.
  The return value is not the same Tensor as the original, but contains the same
  values.  This operation is fast when used on the same device.
  For example:
  >>> a = tf.constant([0.78])
  >>> a_identity = tf.identity(a)
  >>> a.numpy()
  array([0.78], dtype=float32)
  >>> a_identity.numpy()
  array([0.78], dtype=float32)
  Calling `tf.identity` on a variable will make a Tensor that represents the
  value of that variable at the time it is called. This is equivalent to calling
  `<variable>.read_value()`.
  >>> a = tf.Variable(5)
  >>> a_identity = tf.identity(a)
  >>> a.assign_add(1)
  <tf.Variable ... shape=() dtype=int32, numpy=6>
  >>> a.numpy()
  6
  >>> a_identity.numpy()
  5
  This function can also be used to explicitly transfer tensors between devices.
  For example, to transfer a tensor in GPU memory back to host memory, one can
  use:
  >>> with tf.device("/gpu:0"):
  ...   x_on_gpu = tf.constant(1)
  >>> with tf.device("/cpu:0"):
  ...   x_on_cpu = tf.identity(x_on_gpu)
  >>> x_on_cpu.device
  '/job:localhost/replica:0/task:0/device:CPU:0'
  Args:
    input: A `Tensor`, a `Variable`, a `CompositeTensor` or anything that can be
    converted to a tensor using `tf.convert_to_tensor`.
    name: A name for the operation (optional).
  Returns:
    A `Tensor` or CompositeTensor. Has the same type and contents as `input`.
  """
  # Don't expand ResourceVariables, so identity(variable) will return a Tensor.
  if (isinstance(input, composite_tensor.CompositeTensor) and
      not _pywrap_utils.IsResourceVariable(input)):
    return nest.map_structure(identity, input, expand_composites=True)
  if context.executing_eagerly() and not hasattr(input, "graph"):
    # Make sure we get an input with handle data attached from resource
    # variables. Variables have correct handle data when graph building.
    input = ops.convert_to_tensor(input)
  ret = gen_array_ops.identity(input, name=name)
  # Propagate handle data for happier shape inference for resource variables.
  if hasattr(input, "_handle_data"):
    ret._handle_data = input._handle_data  # pylint: disable=protected-access
  return ret
