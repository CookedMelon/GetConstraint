@tf_export("data.experimental.service.ShardingPolicy")
class ShardingPolicy(enum.IntEnum):
  """Specifies how to shard data among tf.data service workers.
  OFF: No sharding will be performed. Each worker produces the entire dataset
  without any sharding. With this mode, the best practice is to shuffle the
  dataset nondeterministically so that workers process the dataset in different
  orders. If workers are restarted or join the cluster mid-job, they will begin
  processing the dataset from the beginning.
  DYNAMIC: The input dataset is dynamically split among workers at runtime. Each
  worker gets the next split when it reads data from the dispatcher. Data is
  produced non-deterministically in this mode. Dynamic sharding works well with
  varying-sized tf.data service clusters, e.g., when you need to auto-scale your
  workers. Dynamic sharding provides at-most once visitation guarantees. No
  examples will be repeated, but some may be missed if a tf.data service worker
  gets restarted while processing a file.
  The following are static sharding policies. The semantics are similar to
  `tf.data.experimental.AutoShardPolicy`. These policies require:
  * The tf.data service cluster is configured with a fixed list of workers
    in DispatcherConfig.
  * Each client only reads from the local tf.data service worker.
  If a worker is restarted while performing static sharding, the worker will
  begin processing its shard again from the beginning.
  FILE: Shards by input files (i.e. each worker will get a fixed set of files to
  process). When this option is selected, make sure that there is at least as
  many files as workers. If there are fewer input files than workers, a runtime
  error will be raised.
  DATA: Shards by elements produced by the dataset. Each worker will process the
  whole dataset and discard the portion that is not for itself. Note that for
  this mode to correctly partition the dataset elements, the dataset needs to
  produce elements in a deterministic order.
  FILE_OR_DATA: Attempts FILE-based sharding, falling back to DATA-based
  sharding on failure.
  HINT: Looks for the presence of `shard(SHARD_HINT, ...)` which is treated as a
  placeholder to replace with `shard(num_workers, worker_index)`.
  """
  # LINT.IfChange(tf_data_service_sharding_policy)
  OFF = 0
  DYNAMIC = 1
  FILE = 2
  DATA = 3
  FILE_OR_DATA = 4
  HINT = 5
  # LINT.ThenChange()
  def _to_proto(self):
    """Converts the policy to ProcessingModeDef proto enum."""
    if self == ShardingPolicy.OFF:
      return data_service_pb2.ProcessingModeDef.OFF
    if self == ShardingPolicy.DYNAMIC:
      return data_service_pb2.ProcessingModeDef.DYNAMIC
    if self == ShardingPolicy.FILE:
      return data_service_pb2.ProcessingModeDef.FILE
    if self == ShardingPolicy.DATA:
      return data_service_pb2.ProcessingModeDef.DATA
    if self == ShardingPolicy.FILE_OR_DATA:
      return data_service_pb2.ProcessingModeDef.FILE_OR_DATA
    if self == ShardingPolicy.HINT:
      return data_service_pb2.ProcessingModeDef.HINT
    raise ValueError(f"Unable to convert sharding policy {self!r} to proto.")
