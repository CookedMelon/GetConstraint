@tf_export("distribute.InputOptions", v1=[])
class InputOptions(
    collections.namedtuple("InputOptions", [
        "experimental_fetch_to_device",
        "experimental_replication_mode",
        "experimental_place_dataset_on_device",
        "experimental_per_replica_buffer_size",
    ])):
  """Run options for `experimental_distribute_dataset(s_from_function)`.
  This can be used to hold some strategy specific configs.
  ```python
  # Setup TPUStrategy
  resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
  tf.config.experimental_connect_to_cluster(resolver)
  tf.tpu.experimental.initialize_tpu_system(resolver)
  strategy = tf.distribute.TPUStrategy(resolver)
  dataset = tf.data.Dataset.range(16)
  distributed_dataset_on_host = (
      strategy.experimental_distribute_dataset(
          dataset,
          tf.distribute.InputOptions(
              experimental_replication_mode=
              experimental_replication_mode.PER_WORKER,
              experimental_place_dataset_on_device=False,
              experimental_per_replica_buffer_size=1)))
  ```
  Attributes:
    experimental_fetch_to_device: Boolean. If True, dataset
      elements will be prefetched to accelerator device memory. When False,
      dataset elements are prefetched to host device memory. Must be False when
      using TPUEmbedding API. experimental_fetch_to_device can only be used
      with experimental_replication_mode=PER_WORKER. Default behavior is same as
      setting it to True.
    experimental_replication_mode: Replication mode for the input function.
      Currently, the InputReplicationMode.PER_REPLICA is only supported with
      tf.distribute.MirroredStrategy.
      experimental_distribute_datasets_from_function.
      The default value is InputReplicationMode.PER_WORKER.
    experimental_place_dataset_on_device: Boolean. Default to False. When True,
      dataset will be placed on the device, otherwise it will remain on the
      host. experimental_place_dataset_on_device=True can only be used with
      experimental_replication_mode=PER_REPLICA
    experimental_per_replica_buffer_size: Integer. Default to 1. Indicates the
      prefetch buffer size in the replica device memory. Users can set it
      to 0 to completely disable prefetching behavior, or a number greater than
      1 to enable larger buffer size. Note that this option is still
      valid with `experimental_fetch_to_device=False`.
  """
  def __new__(cls,
              experimental_fetch_to_device=None,
              experimental_replication_mode=InputReplicationMode.PER_WORKER,
              experimental_place_dataset_on_device=False,
              experimental_per_replica_buffer_size=1):
    if experimental_fetch_to_device is None:
      experimental_fetch_to_device = True
    return super(InputOptions,
                 cls).__new__(cls, experimental_fetch_to_device,
                              experimental_replication_mode,
                              experimental_place_dataset_on_device,
                              experimental_per_replica_buffer_size)
