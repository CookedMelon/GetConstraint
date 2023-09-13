@tf_export("data.experimental.copy_to_device")
def copy_to_device(target_device, source_device="/cpu:0"):
  """A transformation that copies dataset elements to the given `target_device`.
  Args:
    target_device: The name of a device to which elements will be copied.
    source_device: The original device on which `input_dataset` will be placed.
  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.
  """
  def _apply_fn(dataset):
    return _CopyToDeviceDataset(
        dataset, target_device=target_device, source_device=source_device)
  return _apply_fn
