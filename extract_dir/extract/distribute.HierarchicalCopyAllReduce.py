@tf_export("distribute.HierarchicalCopyAllReduce")
class HierarchicalCopyAllReduce(AllReduceCrossDeviceOps):
  """Hierarchical copy all-reduce implementation of CrossDeviceOps.
  It reduces to one GPU along edges in some hierarchy and broadcasts back to
  each GPU along the same path. For the batch API, tensors will be repacked or
  aggregated for more efficient cross-device transportation.
  This is a reduction created for Nvidia DGX-1 which assumes GPUs connects like
  that on DGX-1 machine. If you have different GPU inter-connections, it is
  likely that it would be slower than `tf.distribute.ReductionToOneDevice`.
  For reduces that are not all-reduce, it falls back to
  `tf.distribute.ReductionToOneDevice`.
  Here is how you can use `HierarchicalCopyAllReduce` in
  `tf.distribute.MirroredStrategy`:
  ```
    strategy = tf.distribute.MirroredStrategy(
      cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
  ```
  """
  def __init__(self, num_packs=1):
    """Initializes the object.
    Args:
      num_packs: a non-negative integer. The number of packs to split values
        into. If zero, no packing will be done.
    Raises:
      ValueError if `num_packs` is negative.
    """
    if num_packs < 0:
      raise ValueError(
          "HierarchicalCopy requires num_packs >= 0, but {} is specified"
          .format(num_packs))
    super(HierarchicalCopyAllReduce, self).__init__(
        all_reduce_alg="hierarchical_copy",
        num_packs=num_packs)
