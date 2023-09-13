@tf_export("distribute.InputReplicationMode")
class InputReplicationMode(enum.Enum):
  """Replication mode for input function.
  * `PER_WORKER`: The input function will be called on each worker
    independently, creating as many input pipelines as number of workers.
    Replicas will dequeue from the local Dataset on their worker.
    `tf.distribute.Strategy` doesn't manage any state sharing between such
    separate input pipelines.
  * `PER_REPLICA`: The input function will be called on each replica separately.
    `tf.distribute.Strategy` doesn't manage any state sharing between such
    separate input pipelines.
  """
  PER_WORKER = "PER_WORKER"
  PER_REPLICA = "PER_REPLICA"
