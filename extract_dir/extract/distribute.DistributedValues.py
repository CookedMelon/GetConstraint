@tf_export("distribute.DistributedValues", v1=[])
class DistributedValues(object):
  """Base class for representing distributed values.
  A subclass instance of `tf.distribute.DistributedValues` is created when
  creating variables within a distribution strategy, iterating a
  `tf.distribute.DistributedDataset` or through `tf.distribute.Strategy.run`.
  This base class should never be instantiated directly.
  `tf.distribute.DistributedValues` contains a value per replica. Depending on
  the subclass, the values could either be synced on update, synced on demand,
  or never synced.
  Two representative types of `tf.distribute.DistributedValues` are
  `tf.types.experimental.PerReplica` and `tf.types.experimental.Mirrored`
  values.
  `PerReplica` values exist on the worker devices, with a different value for
  each replica. They are produced by iterating through a distributed dataset
  returned by `tf.distribute.Strategy.experimental_distribute_dataset` (Example
  1, below) and `tf.distribute.Strategy.distribute_datasets_from_function`. They
  are also the typical result returned by `tf.distribute.Strategy.run` (Example
  2).
  `Mirrored` values are like `PerReplica` values, except we know that the value
  on all replicas are the same. `Mirrored` values are kept synchronized by the
  distribution strategy in use, while `PerReplica` values are left
  unsynchronized. `Mirrored` values typically represent model weights. We can
  safely read a `Mirrored` value in a cross-replica context by using the value
  on any replica, while PerReplica values should not be read or manipulated in
  a cross-replica context."
  `tf.distribute.DistributedValues` can be reduced via `strategy.reduce` to
  obtain a single value across replicas (Example 4), used as input into
  `tf.distribute.Strategy.run` (Example 3), or collected to inspect the
  per-replica values using `tf.distribute.Strategy.experimental_local_results`
  (Example 5).
  Example usages:
  1. Created from a `tf.distribute.DistributedDataset`:
  >>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
  >>> dataset = tf.data.Dataset.from_tensor_slices([5., 6., 7., 8.]).batch(2)
  >>> dataset_iterator = iter(strategy.experimental_distribute_dataset(dataset))
  >>> distributed_values = next(dataset_iterator)
  >>> distributed_values
  PerReplica:{
    0: <tf.Tensor: shape=(1,), dtype=float32, numpy=array([5.], dtype=float32)>,
    1: <tf.Tensor: shape=(1,), dtype=float32, numpy=array([6.], dtype=float32)>
  }
  2. Returned by `run`:
  >>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
  >>> @tf.function
  ... def run():
  ...   ctx = tf.distribute.get_replica_context()
  ...   return ctx.replica_id_in_sync_group
  >>> distributed_values = strategy.run(run)
  >>> distributed_values
  PerReplica:{
    0: <tf.Tensor: shape=(), dtype=int32, numpy=0>,
    1: <tf.Tensor: shape=(), dtype=int32, numpy=1>
  }
  3. As input into `run`:
  >>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
  >>> dataset = tf.data.Dataset.from_tensor_slices([5., 6., 7., 8.]).batch(2)
  >>> dataset_iterator = iter(strategy.experimental_distribute_dataset(dataset))
  >>> distributed_values = next(dataset_iterator)
  >>> @tf.function
  ... def run(input):
  ...   return input + 1.0
  >>> updated_value = strategy.run(run, args=(distributed_values,))
  >>> updated_value
  PerReplica:{
    0: <tf.Tensor: shape=(1,), dtype=float32, numpy=array([6.], dtype=float32)>,
    1: <tf.Tensor: shape=(1,), dtype=float32, numpy=array([7.], dtype=float32)>
  }
  4. As input into `reduce`:
  >>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
  >>> dataset = tf.data.Dataset.from_tensor_slices([5., 6., 7., 8.]).batch(2)
  >>> dataset_iterator = iter(strategy.experimental_distribute_dataset(dataset))
  >>> distributed_values = next(dataset_iterator)
  >>> reduced_value = strategy.reduce(tf.distribute.ReduceOp.SUM,
  ...                                 distributed_values,
  ...                                 axis = 0)
  >>> reduced_value
  <tf.Tensor: shape=(), dtype=float32, numpy=11.0>
  5. How to inspect per-replica values locally:
  >>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
  >>> dataset = tf.data.Dataset.from_tensor_slices([5., 6., 7., 8.]).batch(2)
  >>> dataset_iterator = iter(strategy.experimental_distribute_dataset(dataset))
  >>> per_replica_values = strategy.experimental_local_results(
  ...    distributed_values)
  >>> per_replica_values
  (<tf.Tensor: shape=(1,), dtype=float32, numpy=array([5.], dtype=float32)>,
   <tf.Tensor: shape=(1,), dtype=float32, numpy=array([6.], dtype=float32)>)
  """
