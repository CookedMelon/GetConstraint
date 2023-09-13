@tf_export("test.gpu_device_name")
def gpu_device_name():
  """Returns the name of a GPU device if available or a empty string.
  This method should only be used in tests written with `tf.test.TestCase`.
  >>> class MyTest(tf.test.TestCase):
  ...
  ...   def test_add_on_gpu(self):
  ...     if not tf.test.is_built_with_gpu_support():
  ...       self.skipTest("test is only applicable on GPU")
  ...
  ...     with tf.device(tf.test.gpu_device_name()):
  ...       self.assertEqual(tf.math.add(1.0, 2.0), 3.0)
  """
  for x in device_lib.list_local_devices():
    if x.device_type == "GPU":
      return compat.as_str(x.name)
  return ""
