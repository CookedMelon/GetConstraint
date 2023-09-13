@tf_export("device", v1=[])
def device_v2(device_name):
  """Specifies the device for ops created/executed in this context.
  This function specifies the device to be used for ops created/executed in a
  particular context. Nested contexts will inherit and also create/execute
  their ops on the specified device. If a specific device is not required,
  consider not using this function so that a device can be automatically
  assigned.  In general the use of this function is optional. `device_name` can
  be fully specified, as in "/job:worker/task:1/device:cpu:0", or partially
  specified, containing only a subset of the "/"-separated fields. Any fields
  which are specified will override device annotations from outer scopes.
  For example:
  ```python
  with tf.device('/job:foo'):
    # ops created here have devices with /job:foo
    with tf.device('/job:bar/task:0/device:gpu:2'):
      # ops created here have the fully specified device above
    with tf.device('/device:gpu:1'):
      # ops created here have the device '/job:foo/device:gpu:1'
  ```
  Args:
    device_name: The device name to use in the context.
  Returns:
    A context manager that specifies the default device to use for newly
    created ops.
  Raises:
    RuntimeError: If a function is passed in.
  """
  if callable(device_name):
    raise RuntimeError("tf.device does not support functions.")
  return device(device_name)
