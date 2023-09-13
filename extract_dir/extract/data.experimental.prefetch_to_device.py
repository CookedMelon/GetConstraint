@tf_export("data.experimental.prefetch_to_device")
def prefetch_to_device(device, buffer_size=None):
  """A transformation that prefetches dataset values to the given `device`.
  NOTE: Although the transformation creates a `tf.data.Dataset`, the
  transformation must be the final `Dataset` in the input pipeline.
  For example,
  >>> dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
  >>> dataset = dataset.apply(tf.data.experimental.prefetch_to_device("/cpu:0"))
  >>> for element in dataset:
  ...   print(f'Tensor {element} is on device {element.device}')
  Tensor 1 is on device /job:localhost/replica:0/task:0/device:CPU:0
  Tensor 2 is on device /job:localhost/replica:0/task:0/device:CPU:0
  Tensor 3 is on device /job:localhost/replica:0/task:0/device:CPU:0
  Args:
    device: A string. The name of a device to which elements will be prefetched.
    buffer_size: (Optional.) The number of elements to buffer on `device`.
      Defaults to an automatically chosen value.
  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.
  """
  def _apply_fn(dataset):
    return dataset.apply(
        copy_to_device(target_device=device)).prefetch(buffer_size)
  return _apply_fn
