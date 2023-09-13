@tf_export("types.experimental.distributed.Mirrored", v1=[])
class Mirrored(DistributedValues):
  """Holds a distributed value: a map from replica id to synchronized values.
  `Mirrored` values are `tf.distribute.DistributedValues` for which we know that
  the value on all replicas is the same. `Mirrored` values are kept synchronized
  by the distribution strategy in use, while `tf.types.experimental.PerReplica`
  values are left unsynchronized. `Mirrored` values typically represent model
  weights. We can safely read a `Mirrored` value in a cross-replica context by
  using the value on any replica, while `PerReplica` values should not be read
  or manipulated directly by the user in a cross-replica context.
  """
