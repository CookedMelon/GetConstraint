@tf_export("distribute.NcclAllReduce")
class NcclAllReduce(AllReduceCrossDeviceOps):
  """NCCL all-reduce implementation of CrossDeviceOps.
  It uses Nvidia NCCL for all-reduce. For the batch API, tensors will be
  repacked or aggregated for more efficient cross-device transportation.
  For reduces that are not all-reduce, it falls back to
  `tf.distribute.ReductionToOneDevice`.
  Here is how you can use `NcclAllReduce` in `tf.distribute.MirroredStrategy`:
  ```
    strategy = tf.distribute.MirroredStrategy(
      cross_device_ops=tf.distribute.NcclAllReduce())
  ```
  """
  def __init__(self, num_packs=1):
    """Initializes the object.
    Args:
      num_packs: a non-negative integer. The number of packs to split values
        into. If zero, no packing will be done.
    Raises:
      ValueError: if `num_packs` is negative.
    """
    if num_packs < 0:
      raise ValueError(
          "NCCL all-reduce requires num_packs >= 0, but {} is specified".format(
              num_packs))
    super(NcclAllReduce, self).__init__(
        all_reduce_alg="nccl", num_packs=num_packs)
