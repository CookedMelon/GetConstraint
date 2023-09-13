@tf_export("data.experimental.DistributeOptions")
class DistributeOptions(options_lib.OptionsBase):
  """Represents options for distributed data processing.
  You can set the distribution options of a dataset through the
  `experimental_distribute` property of `tf.data.Options`; the property is
  an instance of `tf.data.experimental.DistributeOptions`.
  ```python
  options = tf.data.Options()
  options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
  dataset = dataset.with_options(options)
  ```
  """
  auto_shard_policy = options_lib.create_option(
      name="auto_shard_policy",
      ty=AutoShardPolicy,
      docstring="The type of sharding to use. See "
      "`tf.data.experimental.AutoShardPolicy` for additional information.",
      default_factory=lambda: AutoShardPolicy.AUTO)
  num_devices = options_lib.create_option(
      name="num_devices",
      ty=int,
      docstring=
      "The number of devices attached to this input pipeline. This will be "
      "automatically set by `MultiDeviceIterator`.")
  def _to_proto(self):
    pb = dataset_options_pb2.DistributeOptions()
    pb.auto_shard_policy = AutoShardPolicy._to_proto(self.auto_shard_policy)  # pylint: disable=protected-access
    if self.num_devices is not None:
      pb.num_devices = self.num_devices
    return pb
  def _from_proto(self, pb):
    self.auto_shard_policy = AutoShardPolicy._from_proto(pb.auto_shard_policy)  # pylint: disable=protected-access
    if pb.WhichOneof("optional_num_devices") is not None:
      self.num_devices = pb.num_devices
