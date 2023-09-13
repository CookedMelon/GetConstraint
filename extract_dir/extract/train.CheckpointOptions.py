@tf_export("train.CheckpointOptions")
class CheckpointOptions(object):
  """Options for constructing a Checkpoint.
  Used as the `options` argument to either `tf.train.Checkpoint.save()` or
  `tf.train.Checkpoint.restore()` methods to adjust how variables are
  saved/restored.
  Example: Run IO ops on "localhost" while saving a checkpoint:
  ```
  step = tf.Variable(0, name="step")
  checkpoint = tf.train.Checkpoint(step=step)
  options = tf.train.CheckpointOptions(experimental_io_device="/job:localhost")
  checkpoint.save("/tmp/ckpt", options=options)
  ```
  """
  # Define object attributes in __slots__ for improved memory and performance.
  __slots__ = (
      "experimental_io_device",
      "experimental_enable_async_checkpoint",
      "enable_async",
  )
  @deprecated_args(
      None, "Use enable_async instead", "experimental_enable_async_checkpoint"
  )
  def __init__(
      self,
      experimental_io_device=None,
      experimental_enable_async_checkpoint=False,
      enable_async=False,
  ):
    """Creates an object that stores options for a Checkpoint.
    Args:
      experimental_io_device: string. Applies in a distributed setting.
        Tensorflow device to use to access the filesystem. If `None` (default)
        then for each variable the filesystem is accessed from the CPU:0 device
        of the host where that variable is assigned. If specified, the
        filesystem is instead accessed from that device for all variables.
        This is for example useful if you want to save to a local directory,
        such as "/tmp" when running in a distributed setting. In that case pass
        a device for the host where the "/tmp" directory is accessible.
      experimental_enable_async_checkpoint: bool Type. Deprecated, please use
        the enable_async option.
      enable_async: bool Type. Indicates whether async checkpointing is enabled.
        Default is False, i.e., no async checkpoint.
        Async checkpoint moves the checkpoint file writing off the main thread,
        so that the model can continue to train while the checkpoing file
        writing runs in the background. Async checkpoint reduces TPU device idle
        cycles and speeds up model training process, while memory consumption
        may increase.
    """
    self.experimental_io_device = experimental_io_device
    self.enable_async = experimental_enable_async_checkpoint or enable_async
    self.experimental_enable_async_checkpoint = self.enable_async
