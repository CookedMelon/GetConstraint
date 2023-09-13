@tf_export("distribute.TPUStrategy", v1=[])
class TPUStrategyV2(distribute_lib.Strategy):
  """Synchronous training on TPUs and TPU Pods.
  To construct a TPUStrategy object, you need to run the
  initialization code as below:
  >>> resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
  >>> tf.config.experimental_connect_to_cluster(resolver)
  >>> tf.tpu.experimental.initialize_tpu_system(resolver)
  >>> strategy = tf.distribute.TPUStrategy(resolver)
  While using distribution strategies, the variables created within the
  strategy's scope will be replicated across all the replicas and can be kept in
  sync using all-reduce algorithms.
  To run TF2 programs on TPUs, you can either use `.compile` and
  `.fit` APIs in `tf.keras` with TPUStrategy, or write your own customized
  training loop by calling `strategy.run` directly. Note that
  TPUStrategy doesn't support pure eager execution, so please make sure the
  function passed into `strategy.run` is a `tf.function` or
  `strategy.run` is called inside a `tf.function` if eager
  behavior is enabled. See more details in https://www.tensorflow.org/guide/tpu.
  `distribute_datasets_from_function` and
  `experimental_distribute_dataset` APIs can be used to distribute the dataset
  across the TPU workers when writing your own training loop. If you are using
  `fit` and `compile` methods available in `tf.keras.Model`, then Keras will
  handle the distribution for you.
  An example of writing customized training loop on TPUs:
  >>> with strategy.scope():
  ...   model = tf.keras.Sequential([
  ...     tf.keras.layers.Dense(2, input_shape=(5,)),
  ...   ])
  ...   optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
  >>> def dataset_fn(ctx):
  ...   x = np.random.random((2, 5)).astype(np.float32)
  ...   y = np.random.randint(2, size=(2, 1))
  ...   dataset = tf.data.Dataset.from_tensor_slices((x, y))
  ...   return dataset.repeat().batch(1, drop_remainder=True)
  >>> dist_dataset = strategy.distribute_datasets_from_function(
  ...     dataset_fn)
  >>> iterator = iter(dist_dataset)
  >>> @tf.function()
  ... def train_step(iterator):
  ...
  ...   def step_fn(inputs):
  ...     features, labels = inputs
  ...     with tf.GradientTape() as tape:
  ...       logits = model(features, training=True)
  ...       loss = tf.keras.losses.sparse_categorical_crossentropy(
  ...           labels, logits)
  ...
  ...     grads = tape.gradient(loss, model.trainable_variables)
  ...     optimizer.apply_gradients(zip(grads, model.trainable_variables))
  ...
  ...   strategy.run(step_fn, args=(next(iterator),))
  >>> train_step(iterator)
  For the advanced use cases like model parallelism, you can set
  `experimental_device_assignment` argument when creating TPUStrategy to specify
  number of replicas and number of logical devices. Below is an example to
  initialize TPU system with 2 logical devices and 1 replica.
  >>> resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
  >>> tf.config.experimental_connect_to_cluster(resolver)
  >>> topology = tf.tpu.experimental.initialize_tpu_system(resolver)
  >>> device_assignment = tf.tpu.experimental.DeviceAssignment.build(
  ...     topology,
  ...     computation_shape=[1, 1, 1, 2],
  ...     num_replicas=1)
  >>> strategy = tf.distribute.TPUStrategy(
  ...     resolver, experimental_device_assignment=device_assignment)
  Then you can run a `tf.add` operation only on logical device 0.
  >>> @tf.function()
  ... def step_fn(inputs):
  ...   features, _ = inputs
  ...   output = tf.add(features, features)
  ...
  ...   # Add operation will be executed on logical device 0.
  ...   output = strategy.experimental_assign_to_logical_device(output, 0)
  ...   return output
  >>> dist_dataset = strategy.distribute_datasets_from_function(
  ...     dataset_fn)
  >>> iterator = iter(dist_dataset)
  >>> strategy.run(step_fn, args=(next(iterator),))
  `experimental_spmd_xla_partitioning` enables the experimental XLA SPMD feature
  for model parallelism. This flag can reduce the compilation time and HBM
  requirements. When running in this mode, every input tensor must either be
  partitioned (via `strategy.experimental_split_to_logical_devices`) or fully
  replicated (via `strategy.experimental_replicate_to_logical_devices`) to all
  logical devices. And calling `strategy.experimental_assign_to_logical_device`
  will result in a ValueError in this mode.
  """
  def __init__(self,
               tpu_cluster_resolver=None,
               experimental_device_assignment=None,
               experimental_spmd_xla_partitioning=False):
    """Synchronous training in TPU donuts or Pods.
    Args:
      tpu_cluster_resolver: A
        `tf.distribute.cluster_resolver.TPUClusterResolver` instance, which
        provides information about the TPU cluster. If None, it will assume
        running on a local TPU worker.
      experimental_device_assignment: Optional
        `tf.tpu.experimental.DeviceAssignment` to specify the placement of
        replicas on the TPU cluster.
      experimental_spmd_xla_partitioning: If True, enable the SPMD (Single
        Program Multiple Data) mode in XLA compiler. This flag only affects the
        performance of XLA compilation and the HBM requirement of the compiled
        TPU program. Ceveat: if this flag is True, calling
        `tf.distribute.TPUStrategy.experimental_assign_to_logical_device` will
        result in a ValueError.
    """
    super(TPUStrategyV2, self).__init__(
        TPUExtended(
            self,
            tpu_cluster_resolver,
            device_assignment=experimental_device_assignment,
            use_spmd_for_xla_partitioning=experimental_spmd_xla_partitioning,
            enable_data_reorder=experimental_device_assignment is not None,
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
  def run(self, fn, args=(), kwargs=None, options=None):
    """Run the computation defined by `fn` on each TPU replica.
    Executes ops specified by `fn` on each replica. If `args` or `kwargs` have
    `tf.distribute.DistributedValues`, such as those produced by a
    `tf.distribute.DistributedDataset` from
    `tf.distribute.Strategy.experimental_distribute_dataset` or
    `tf.distribute.Strategy.distribute_datasets_from_function`,
    when `fn` is executed on a particular replica, it will be executed with the
    component of `tf.distribute.DistributedValues` that correspond to that
    replica.
    `fn` may call `tf.distribute.get_replica_context()` to access members such
    as `all_reduce`.
    All arguments in `args` or `kwargs` should either be nest of tensors or
    `tf.distribute.DistributedValues` containing tensors or composite tensors.
    Example usage:
    >>> resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
    >>> tf.config.experimental_connect_to_cluster(resolver)
    >>> tf.tpu.experimental.initialize_tpu_system(resolver)
    >>> strategy = tf.distribute.TPUStrategy(resolver)
    >>> @tf.function
    ... def run():
    ...   def value_fn(value_context):
    ...     return value_context.num_replicas_in_sync
    ...   distributed_values = (
    ...       strategy.experimental_distribute_values_from_function(value_fn))
    ...   def replica_fn(input):
    ...     return input * 2
    ...   return strategy.run(replica_fn, args=(distributed_values,))
    >>> result = run()
    Args:
      fn: The function to run. The output must be a `tf.nest` of `Tensor`s.
      args: (Optional) Positional arguments to `fn`.
      kwargs: (Optional) Keyword arguments to `fn`.
      options: (Optional) An instance of `tf.distribute.RunOptions` specifying
        the options to run `fn`.
    Returns:
      Merged return value of `fn` across replicas. The structure of the return
      value is the same as the return value from `fn`. Each element in the
      structure can either be `tf.distribute.DistributedValues`, `Tensor`
      objects, or `Tensor`s (for example, if running on a single replica).
    """
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
    `tf.distribute.TPUStrategy` provides the associated
    `tf.distribute.cluster_resolver.ClusterResolver`. If the user provides one
    in `__init__`, that instance is returned; if the user does not, a default
    `tf.distribute.cluster_resolver.TPUClusterResolver` is provided.
    """
    return self.extended._tpu_cluster_resolver  # pylint: disable=protected-access
  def experimental_assign_to_logical_device(self, tensor, logical_device_id):
    """Adds annotation that `tensor` will be assigned to a logical device.
    This adds an annotation to `tensor` specifying that operations on
    `tensor` will be invoked on logical core device id `logical_device_id`.
    When model parallelism is used, the default behavior is that all ops
    are placed on zero-th logical device.
    ```python
    # Initializing TPU system with 2 logical devices and 4 replicas.
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
    tf.config.experimental_connect_to_cluster(resolver)
    topology = tf.tpu.experimental.initialize_tpu_system(resolver)
    device_assignment = tf.tpu.experimental.DeviceAssignment.build(
        topology,
        computation_shape=[1, 1, 1, 2],
        num_replicas=4)
    strategy = tf.distribute.TPUStrategy(
        resolver, experimental_device_assignment=device_assignment)
    iterator = iter(inputs)
    @tf.function()
    def step_fn(inputs):
      output = tf.add(inputs, inputs)
      # Add operation will be executed on logical device 0.
      output = strategy.experimental_assign_to_logical_device(output, 0)
      return output
    strategy.run(step_fn, args=(next(iterator),))
    ```
    Args:
      tensor: Input tensor to annotate.
      logical_device_id: Id of the logical core to which the tensor will be
        assigned.
    Raises:
      ValueError: The logical device id presented is not consistent with total
      number of partitions specified by the device assignment or the TPUStrategy
      is constructed with `experimental_spmd_xla_partitioning=True`.
    Returns:
      Annotated tensor with identical value as `tensor`.
    """
    if self.extended._use_spmd_for_xla_partitioning:  # pylint: disable=protected-access
      raise ValueError(
          "Cannot assign a tensor to a logical device in SPMD mode. To disable "
          "SPMD, Please construct the TPUStrategy with "
          "`experimental_spmd_xla_partitioning=False`")
    num_logical_devices_per_replica = self.extended._tpu_devices.shape[1]  # pylint: disable=protected-access
    if (logical_device_id < 0 or
        logical_device_id >= num_logical_devices_per_replica):
      raise ValueError("`logical_core_id` to assign must be lower then total "
                       "number of logical devices per replica. Received "
                       "logical device id {} but there are only total of {} "
                       "logical devices in replica.".format(
                           logical_device_id, num_logical_devices_per_replica))
    return xla_sharding.assign_device(
        tensor, logical_device_id, use_sharding_op=True)
  def experimental_split_to_logical_devices(self, tensor, partition_dimensions):
    """Adds annotation that `tensor` will be split across logical devices.
    This adds an annotation to tensor `tensor` specifying that operations on
    `tensor` will be split among multiple logical devices. Tensor `tensor` will
    be split across dimensions specified by `partition_dimensions`.
    The dimensions of `tensor` must be divisible by corresponding value in
    `partition_dimensions`.
    For example, for system with 8 logical devices, if `tensor` is an image
    tensor with shape (batch_size, width, height, channel) and
    `partition_dimensions` is [1, 2, 4, 1], then `tensor` will be split
    2 in width dimension and 4 way in height dimension and the split
    tensor values will be fed into 8 logical devices.
    ```python
    # Initializing TPU system with 8 logical devices and 1 replica.
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
    tf.config.experimental_connect_to_cluster(resolver)
    topology = tf.tpu.experimental.initialize_tpu_system(resolver)
    device_assignment = tf.tpu.experimental.DeviceAssignment.build(
        topology,
        computation_shape=[1, 2, 2, 2],
        num_replicas=1)
    # Construct the TPUStrategy. Since we are going to split the image across
    # logical devices, here we set `experimental_spmd_xla_partitioning=True`
    # so that the partitioning can be compiled in SPMD mode, which usually
    # results in faster compilation and smaller HBM requirement if the size of
    # input and activation tensors are much bigger than that of the model
    # parameters. Note that this flag is suggested but not a hard requirement
    # for `experimental_split_to_logical_devices`.
    strategy = tf.distribute.TPUStrategy(
        resolver, experimental_device_assignment=device_assignment,
        experimental_spmd_xla_partitioning=True)
    iterator = iter(inputs)
    @tf.function()
    def step_fn(inputs):
      inputs = strategy.experimental_split_to_logical_devices(
        inputs, [1, 2, 4, 1])
      # model() function will be executed on 8 logical devices with `inputs`
      # split 2 * 4  ways.
      output = model(inputs)
      return output
    strategy.run(step_fn, args=(next(iterator),))
    ```
    Args:
      tensor: Input tensor to annotate.
      partition_dimensions: An unnested list of integers with the size equal to
        rank of `tensor` specifying how `tensor` will be partitioned. The
        product of all elements in `partition_dimensions` must be equal to the
        total number of logical devices per replica.
    Raises:
      ValueError: 1) If the size of partition_dimensions does not equal to rank
        of `tensor` or 2) if product of elements of `partition_dimensions` does
        not match the number of logical devices per replica defined by the
        implementing DistributionStrategy's device specification or
        3) if a known size of `tensor` is not divisible by corresponding
        value in `partition_dimensions`.
    Returns:
      Annotated tensor with identical value as `tensor`.
    """
    num_logical_devices_per_replica = self.extended._tpu_devices.shape[1]  # pylint: disable=protected-access
    num_partition_splits = np.prod(partition_dimensions)
    input_shape = tensor.shape
    tensor_rank = len(input_shape)
    if tensor_rank != len(partition_dimensions):
      raise ValueError("Length of `partition_dimensions` must equal to the "
                       "rank of `tensor.shape` ({}). Received "
                       "len(partition_dimensions)={}.".format(
                           tensor_rank, len(partition_dimensions)))
    for dim_index, dim_size in enumerate(input_shape):
      if dim_size is None:
        continue
      split_size = partition_dimensions[dim_index]
      if dim_size % split_size != 0:
        raise ValueError("Tensor shape at `partition_dimensions[{}]` must be "
                         "divisible by corresponding value specified "
                         "by `partition_dimensions` ({}). Received: {}.".format(
                             dim_index, split_size, dim_size))
    if num_partition_splits != num_logical_devices_per_replica:
      raise ValueError(
          "The product of `partition_dimensions` should be the same as the "
          "number of logical devices (={}). Received `partition_dimensions`={},"
          "and their product is {}.".format(num_logical_devices_per_replica,
                                            partition_dimensions,
                                            num_partition_splits))
    tile_assignment = np.arange(num_partition_splits).reshape(
        partition_dimensions)
    return xla_sharding.tile(tensor, tile_assignment, use_sharding_op=True)
  def experimental_replicate_to_logical_devices(self, tensor):
    """Adds annotation that `tensor` will be replicated to all logical devices.
    This adds an annotation to tensor `tensor` specifying that operations on
    `tensor` will be invoked on all logical devices.
    ```python
    # Initializing TPU system with 2 logical devices and 4 replicas.
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
    tf.config.experimental_connect_to_cluster(resolver)
    topology = tf.tpu.experimental.initialize_tpu_system(resolver)
    device_assignment = tf.tpu.experimental.DeviceAssignment.build(
        topology,
        computation_shape=[1, 1, 1, 2],
        num_replicas=4)
    strategy = tf.distribute.TPUStrategy(
        resolver, experimental_device_assignment=device_assignment)
    iterator = iter(inputs)
    @tf.function()
    def step_fn(inputs):
      images, labels = inputs
      images = strategy.experimental_split_to_logical_devices(
        inputs, [1, 2, 4, 1])
      # model() function will be executed on 8 logical devices with `inputs`
      # split 2 * 4  ways.
      output = model(inputs)
      # For loss calculation, all logical devices share the same logits
      # and labels.
      labels = strategy.experimental_replicate_to_logical_devices(labels)
      output = strategy.experimental_replicate_to_logical_devices(output)
      loss = loss_fn(labels, output)
      return loss
    strategy.run(step_fn, args=(next(iterator),))
    ```
    Args:
      tensor: Input tensor to annotate.
    Returns:
      Annotated tensor with identical value as `tensor`.
    """
    return xla_sharding.replicate(tensor, use_sharding_op=True)
