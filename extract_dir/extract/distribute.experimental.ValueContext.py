@tf_export("distribute.experimental.ValueContext", v1=[])
class ValueContext(object):
  """A class wrapping information needed by a distribute function.
  This is a context class that is passed to the `value_fn` in
  `strategy.experimental_distribute_values_from_function` and contains
  information about the compute replicas. The `num_replicas_in_sync` and
  `replica_id` can be used to customize the value on each replica.
  Example usage:
  1.  Directly constructed.
      >>> def value_fn(context):
      ...   return context.replica_id_in_sync_group/context.num_replicas_in_sync
      >>> context = tf.distribute.experimental.ValueContext(
      ...   replica_id_in_sync_group=2, num_replicas_in_sync=4)
      >>> per_replica_value = value_fn(context)
      >>> per_replica_value
      0.5
  2.  Passed in by `experimental_distribute_values_from_function`.  {: value=2}
      >>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
      >>> def value_fn(value_context):
      ...   return value_context.num_replicas_in_sync
      >>> distributed_values = (
      ...      strategy.experimental_distribute_values_from_function(
      ...        value_fn))
      >>> local_result = strategy.experimental_local_results(distributed_values)
      >>> local_result
      (2, 2)
  """
  __slots__ = ["_replica_id_in_sync_group", "_num_replicas_in_sync"]
  def __init__(self,
               replica_id_in_sync_group=0,
               num_replicas_in_sync=1):
    """Initializes an ValueContext object.
    Args:
      replica_id_in_sync_group: the current replica_id, should be an int in
        [0,`num_replicas_in_sync`).
      num_replicas_in_sync: the number of replicas that are in sync.
    """
    self._replica_id_in_sync_group = replica_id_in_sync_group
    self._num_replicas_in_sync = num_replicas_in_sync
  @property
  def num_replicas_in_sync(self):
    """Returns the number of compute replicas in sync."""
    return self._num_replicas_in_sync
  @property
  def replica_id_in_sync_group(self):
    """Returns the replica ID."""
    return self._replica_id_in_sync_group
  def __str__(self):
    return (("tf.distribute.ValueContext(replica id {}, "
             " total replicas in sync: ""{})")
            .format(self.replica_id_in_sync_group, self.num_replicas_in_sync))
