@tf_export("config.LogicalDevice")
class LogicalDevice(
    collections.namedtuple("LogicalDevice", ["name", "device_type"])):
  """Abstraction for a logical device initialized by the runtime.
  A `tf.config.LogicalDevice` corresponds to an initialized logical device on a
  `tf.config.PhysicalDevice` or a remote device visible to the cluster. Tensors
  and operations can be placed on a specific logical device by calling
  `tf.device` with a specified `tf.config.LogicalDevice`.
  Fields:
    name: The fully qualified name of the device. Can be used for Op or function
      placement.
    device_type: String declaring the type of device such as "CPU" or "GPU".
  """
  pass
