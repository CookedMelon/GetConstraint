@tf_export("tpu.XLAOptions")
class XLAOptions(
    collections.namedtuple("XLAOptions", [
        "use_spmd_for_xla_partitioning",
        "enable_xla_dynamic_padder",
    ])):
  """XLA compilation options.
  Attributes:
    use_spmd_for_xla_partitioning: Boolean. Whether to use XLA's SPMD
      partitioner instead of MPMD partitioner when compiler partitioning is
      requested.
    enable_xla_dynamic_padder: Boolean. Whether to enable XLA dynamic padder
      infrastructure to handle dynamic shapes inputs inside XLA. True by
      default. Disabling this may cause correctness issues with dynamic shapes
      inputs, as XLA will just assume the inputs are with padded shapes. However
      users can optionally set it to False to improve device time if masking is
      already handled in the user side.
  """
  def __new__(cls,
              use_spmd_for_xla_partitioning=True,
              enable_xla_dynamic_padder=True):
    return super(XLAOptions, cls).__new__(cls, use_spmd_for_xla_partitioning,
                                          enable_xla_dynamic_padder)
