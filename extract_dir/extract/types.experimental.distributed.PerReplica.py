@tf_export("types.experimental.distributed.PerReplica", v1=[])
class PerReplica(DistributedValues):
  """Holds a distributed value: a map from replica id to unsynchronized values.
  `PerReplica` values exist on the worker devices, with a different value for
  each replica. They can be produced many ways, often by iterating through a
  distributed dataset returned by
  `tf.distribute.Strategy.experimental_distribute_dataset` and
  `tf.distribute.Strategy.distribute_datasets_from_function`. They are also the
  typical result returned by `tf.distribute.Strategy.run`.
  """
