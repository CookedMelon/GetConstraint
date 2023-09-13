@tf_export("config.PhysicalDevice")
class PhysicalDevice(
    collections.namedtuple("PhysicalDevice", ["name", "device_type"])):
  """Abstraction for a locally visible physical device.
  TensorFlow can utilize various devices such as the CPU or multiple GPUs
  for computation. Before initializing a local device for use, the user can
  customize certain properties of the device such as it's visibility or memory
  configuration.
  Once a visible `tf.config.PhysicalDevice` is initialized one or more
  `tf.config.LogicalDevice` objects are created. Use
  `tf.config.set_visible_devices` to configure the visibility of a physical
  device and `tf.config.set_logical_device_configuration` to configure multiple
  `tf.config.LogicalDevice` objects for a `tf.config.PhysicalDevice`. This is
  useful when separation between models is needed or to simulate a multi-device
  environment.
  Fields:
    name: Unique identifier for device.
    device_type: String declaring the type of device such as "CPU" or "GPU".
  """
  pass
