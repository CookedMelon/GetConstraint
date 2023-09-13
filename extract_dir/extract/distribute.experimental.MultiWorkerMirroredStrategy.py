@tf_export("distribute.experimental.MultiWorkerMirroredStrategy", v1=[])
class _CollectiveAllReduceStrategyExperimental(
    CollectiveAllReduceStrategy,
    metaclass=_CollectiveAllReduceStrategyExperimentalMeta):
  __doc__ = CollectiveAllReduceStrategy.__doc__
  @deprecation.deprecated(
      None, "use distribute.MultiWorkerMirroredStrategy instead")
  def __init__(self,
               communication=collective_util.CommunicationImplementation.AUTO,
               cluster_resolver=None):
    """Creates the strategy.
    Args:
      communication: optional
        `tf.distribute.experimental.CommunicationImplementation`. This is a hint
        on the preferred collective communication implementation. Possible
        values include `AUTO`, `RING`, and `NCCL`.
      cluster_resolver: optional
        `tf.distribute.cluster_resolver.ClusterResolver`. If `None`,
        `tf.distribute.cluster_resolver.TFConfigClusterResolver` is used.
    """
    communication_options = collective_util.Options(
        implementation=communication)
    super(_CollectiveAllReduceStrategyExperimental,
          self).__init__(cluster_resolver, communication_options)
  @classmethod
  def _from_local_devices(
      cls,
      devices,
      communication=collective_util.CommunicationImplementation.AUTO):
    """A convenience method to create an object with a list of devices."""
    obj = cls(communication)
    obj.extended._initialize_local(TFConfigClusterResolver(), devices=devices)  # pylint: disable=protected-access
    return obj
