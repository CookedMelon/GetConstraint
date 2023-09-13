@tf_export("distribute.experimental.TPUStrategy", v1=[])
@deprecation.deprecated_endpoints("distribute.experimental.TPUStrategy")
class TPUStrategy(distribute_lib.Strategy):
  """Synchronous training on TPUs and TPU Pods.
  To construct a TPUStrategy object, you need to run the
  initialization code as below:
  >>> resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
  >>> tf.config.experimental_connect_to_cluster(resolver)
  >>> tf.tpu.experimental.initialize_tpu_system(resolver)
  >>> strategy = tf.distribute.experimental.TPUStrategy(resolver)
  While using distribution strategies, the variables created within the
  strategy's scope will be replicated across all the replicas and can be kept in
  sync using all-reduce algorithms.
  To run TF2 programs on TPUs, you can either use `.compile` and
  `.fit` APIs in `tf.keras` with TPUStrategy, or write your own customized
  training loop by calling `strategy.run` directly. Note that
  TPUStrategy doesn't support pure eager execution, so please make sure the
  function passed into `strategy.run` is a `tf.function` or
  `strategy.run` is called inside a `tf.function` if eager
  behavior is enabled.
  """
  def __init__(self,
               tpu_cluster_resolver=None,
               device_assignment=None):
    """Synchronous training in TPU donuts or Pods.
    Args:
      tpu_cluster_resolver: A tf.distribute.cluster_resolver.TPUClusterResolver,
        which provides information about the TPU cluster.
      device_assignment: Optional `tf.tpu.experimental.DeviceAssignment` to
        specify the placement of replicas on the TPU cluster.
    """
    logging.warning(
        "`tf.distribute.experimental.TPUStrategy` is deprecated, please use "
        "the non-experimental symbol `tf.distribute.TPUStrategy` instead.")
    super(TPUStrategy, self).__init__(
        TPUExtended(
            self,
            tpu_cluster_resolver,
            device_assignment=device_assignment,
            enable_data_reorder=device_assignment is not None,
        )
    )
    distribute_lib.distribution_strategy_gauge.get_cell("V2").set("TPUStrategy")
    distribute_lib.distribution_strategy_replica_gauge.get_cell(
        "num_workers").set(self.extended.num_hosts)
    distribute_lib.distribution_strategy_replica_gauge.get_cell(
        "num_replicas_per_worker").set(self.extended.num_replicas_per_host)
    # Packed variable is used to reduce the overhead of function execution.
    # For a DistributedVariable, only one variable handle is captured into a
    # function graph. It's only supported in eager mode.
    self._enable_packed_variable_in_eager_mode = True
  # TODO(cjfj): Modify `_call_for_each_replica` in `TPUExtended` such that this
  # can use the default implementation.
  # This implementation runs a single step. It does not use infeed or outfeed.
  def run(self, fn, args=(), kwargs=None, options=None):
    """See base class."""
    validate_run_function(fn)
    fn, args, kwargs = _maybe_partial_apply_variables(fn, args, kwargs)
    # Note: the target function is converted to graph even when in Eager mode,
    # so autograph is on by default here.
    fn = autograph.tf_convert(fn, autograph_ctx.control_status_ctx())
    options = options or distribute_lib.RunOptions()
    return self.extended.tpu_run(fn, args, kwargs, options)
  @property
  def cluster_resolver(self):
    """Returns the cluster resolver associated with this strategy.
    `tf.distribute.experimental.TPUStrategy` provides the
    associated `tf.distribute.cluster_resolver.ClusterResolver`. If the user
    provides one in `__init__`, that instance is returned; if the user does
    not, a default
    `tf.distribute.cluster_resolver.TPUClusterResolver` is provided.
    """
    return self.extended._tpu_cluster_resolver  # pylint: disable=protected-access
