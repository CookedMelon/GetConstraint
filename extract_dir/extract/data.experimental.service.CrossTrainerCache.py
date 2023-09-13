@tf_export("data.experimental.service.CrossTrainerCache")
class CrossTrainerCache:
  """Options related to the tf.data service cross trainer cache.
  This is used to enable cross-trainer cache when distributing a dataset. For
  example:
  ```
  dataset = dataset.apply(tf.data.experimental.service.distribute(
      processing_mode=tf.data.experimental.service.ShardingPolicy.OFF,
      service=FLAGS.tf_data_service_address,
      job_name="job",
      cross_trainer_cache=data_service_ops.CrossTrainerCache(
          trainer_id=trainer_id())))
  ```
  For more details, refer to
  https://www.tensorflow.org/api_docs/python/tf/data/experimental/service#sharing_tfdata_service_with_concurrent_trainers.
  """
  def __init__(self, trainer_id):
    """Constructs a CrossTrainerCache.
    Args:
      trainer_id: Each training job has a unique ID. Once a job has consumed
      data, the data remains in the cache and is re-used by jobs with different
      `trainer_id`s. Requests with the same `trainer_id` do not re-use data.
    Raises:
      ValueError if `trainer_id` is empty.
    """
    if not trainer_id:
      raise ValueError(
          "tf.data service cross-trainer cache requires a non-empty trainer ID."
      )
    self.trainer_id = trainer_id
  def _to_proto(self):
    return data_service_pb2.CrossTrainerCacheOptions(trainer_id=self.trainer_id)
