@tf_export("debugging.set_log_device_placement")
def set_log_device_placement(enabled):
  """Turns logging for device placement decisions on or off.
  Operations execute on a particular device, producing and consuming tensors on
  that device. This may change the performance of the operation or require
  TensorFlow to copy data to or from an accelerator, so knowing where operations
  execute is useful for debugging performance issues.
  For more advanced profiling, use the [TensorFlow
  profiler](https://www.tensorflow.org/guide/profiler).
  Device placement for operations is typically controlled by a `tf.device`
  scope, but there are exceptions, for example operations on a `tf.Variable`
  which follow the initial placement of the variable. Turning off soft device
  placement (with `tf.config.set_soft_device_placement`) provides more explicit
  control.
  >>> tf.debugging.set_log_device_placement(True)
  >>> tf.ones([])
  >>> # [...] op Fill in device /job:localhost/replica:0/task:0/device:GPU:0
  >>> with tf.device("CPU"):
  ...  tf.ones([])
  >>> # [...] op Fill in device /job:localhost/replica:0/task:0/device:CPU:0
  >>> tf.debugging.set_log_device_placement(False)
  Turning on `tf.debugging.set_log_device_placement` also logs the placement of
  ops inside `tf.function` when the function is called.
  Args:
    enabled: Whether to enabled device placement logging.
  """
  context().log_device_placement = enabled
