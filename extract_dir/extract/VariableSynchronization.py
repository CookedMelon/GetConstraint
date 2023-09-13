@tf_export("VariableSynchronization")
class VariableSynchronization(enum.Enum):
  """Indicates when a distributed variable will be synced.
  * `AUTO`: Indicates that the synchronization will be determined by the current
    `DistributionStrategy` (eg. With `MirroredStrategy` this would be
    `ON_WRITE`).
  * `NONE`: Indicates that there will only be one copy of the variable, so
    there is no need to sync.
  * `ON_WRITE`: Indicates that the variable will be updated across devices
    every time it is written.
  * `ON_READ`: Indicates that the variable will be aggregated across devices
    when it is read (eg. when checkpointing or when evaluating an op that uses
    the variable).
    Example:
  >>> temp_grad=[tf.Variable([0.], trainable=False,
  ...                      synchronization=tf.VariableSynchronization.ON_READ,
  ...                      aggregation=tf.VariableAggregation.MEAN
  ...                      )]
  """
  AUTO = 0
  NONE = 1
  ON_WRITE = 2
  ON_READ = 3
