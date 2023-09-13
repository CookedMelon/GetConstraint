@tf_export("data.experimental.AutoShardPolicy")
class AutoShardPolicy(enum.IntEnum):
  """Represents the type of auto-sharding to use.
  OFF: No sharding will be performed.
  AUTO: Attempts FILE-based sharding, falling back to DATA-based sharding.
  FILE: Shards by input files (i.e. each worker will get a set of files to
  process). When this option is selected, make sure that there is at least as
  many files as workers. If there are fewer input files than workers, a runtime
  error will be raised.
  DATA: Shards by elements produced by the dataset. Each worker will process the
  whole dataset and discard the portion that is not for itself. Note that for
  this mode to correctly partitions the dataset elements, the dataset needs to
  produce elements in a deterministic order.
  HINT: Looks for the presence of `shard(SHARD_HINT, ...)` which is treated as a
  placeholder to replace with `shard(num_workers, worker_index)`.
  """
  # LINT.IfChange
  OFF = -1
  AUTO = 0
  FILE = 1
  DATA = 2
  HINT = 3
  # LINT.ThenChange(//tensorflow/python/data/experimental/ops/data_service_ops.py:tf_data_service_sharding_policy)
  @classmethod
  def _to_proto(cls, obj):
    """Convert enum to proto."""
    if obj == cls.OFF:
      return dataset_options_pb2.AutoShardPolicy.OFF
    if obj == cls.FILE:
      return dataset_options_pb2.AutoShardPolicy.FILE
    if obj == cls.DATA:
      return dataset_options_pb2.AutoShardPolicy.DATA
    if obj == cls.AUTO:
      return dataset_options_pb2.AutoShardPolicy.AUTO
    if obj == cls.HINT:
      return dataset_options_pb2.AutoShardPolicy.HINT
    raise ValueError(
        f"Invalid `obj.` Supported values include `OFF`, `FILE`, `DATA`,"
        f"`AUTO`, and `HINT`. Got {obj.name}."
    )
  @classmethod
  def _from_proto(cls, pb):
    """Convert proto to enum."""
    if pb == dataset_options_pb2.AutoShardPolicy.OFF:
      return cls.OFF
    if pb == dataset_options_pb2.AutoShardPolicy.FILE:
      return cls.FILE
    if pb == dataset_options_pb2.AutoShardPolicy.DATA:
      return cls.DATA
    if pb == dataset_options_pb2.AutoShardPolicy.AUTO:
      return cls.AUTO
    if pb == dataset_options_pb2.AutoShardPolicy.HINT:
      return cls.HINT
    raise ValueError(
        f"Invalid `pb.` Supported values include `OFF`, `FILE`, `DATA`,"
        f"`AUTO`, and `HINT`. Got {pb}."
    )
