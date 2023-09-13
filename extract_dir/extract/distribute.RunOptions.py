@tf_export("distribute.RunOptions")
class RunOptions(
    collections.namedtuple("RunOptions", [
        "experimental_enable_dynamic_batch_size",
        "experimental_bucketizing_dynamic_shape",
        "experimental_xla_options",
    ])):
  """Run options for `strategy.run`.
  This can be used to hold some strategy specific configs.
  Attributes:
    experimental_enable_dynamic_batch_size: Boolean. Only applies to
      TPUStrategy. Default to True. If True, TPUStrategy will enable dynamic
      padder to support dynamic batch size for the inputs. Otherwise only static
      shape inputs are allowed.
    experimental_bucketizing_dynamic_shape: Boolean. Only applies to
      TPUStrategy. Default to False. If True, TPUStrategy will automatic
      bucketize inputs passed into `run` if the input shape is
      dynamic. This is a performance optimization to reduce XLA recompilation,
      which should not have impact on correctness.
    experimental_xla_options: A `tf.tpu.XLAOptions` instance. Only applies to
      TPUStrategy. Controls the XLA compiling options on TPUs. Default to None.
  """
  def __new__(cls,
              experimental_enable_dynamic_batch_size=True,
              experimental_bucketizing_dynamic_shape=False,
              experimental_xla_options=None):
    return super(RunOptions,
                 cls).__new__(cls, experimental_enable_dynamic_batch_size,
                              experimental_bucketizing_dynamic_shape,
                              experimental_xla_options)
