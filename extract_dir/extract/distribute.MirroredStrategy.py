@tf_export("distribute.MirroredStrategy", v1=[])  # pylint: disable=g-classes-have-attributes
class MirroredStrategy(distribute_lib.Strategy):
  """Synchronous training across multiple replicas on one machine.
  This strategy is typically used for training on one
  machine with multiple GPUs. For TPUs, use
  `tf.distribute.TPUStrategy`. To use `MirroredStrategy` with multiple workers,
  please refer to `tf.distribute.experimental.MultiWorkerMirroredStrategy`.
  For example, a variable created under a `MirroredStrategy` is a
  `MirroredVariable`. If no devices are specified in the constructor argument of
  the strategy then it will use all the available GPUs. If no GPUs are found, it
  will use the available CPUs. Note that TensorFlow treats all CPUs on a
  machine as a single device, and uses threads internally for parallelism.
  >>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
  >>> with strategy.scope():
  ...   x = tf.Variable(1.)
  >>> x
  MirroredVariable:{
    0: <tf.Variable ... shape=() dtype=float32, numpy=1.0>,
    1: <tf.Variable ... shape=() dtype=float32, numpy=1.0>
  }
  While using distribution strategies, all the variable creation should be done
  within the strategy's scope. This will replicate the variables across all the
  replicas and keep them in sync using an all-reduce algorithm.
  Variables created inside a `MirroredStrategy` which is wrapped with a
  `tf.function` are still `MirroredVariables`.
  >>> x = []
  >>> @tf.function  # Wrap the function with tf.function.
  ... def create_variable():
  ...   if not x:
  ...     x.append(tf.Variable(1.))
  ...   return x[0]
  >>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
  >>> with strategy.scope():
  ...   _ = create_variable()
  ...   print(x[0])
  MirroredVariable:{
    0: <tf.Variable ... shape=() dtype=float32, numpy=1.0>,
    1: <tf.Variable ... shape=() dtype=float32, numpy=1.0>
  }
  `experimental_distribute_dataset` can be used to distribute the dataset across
  the replicas when writing your own training loop. If you are using `.fit` and
  `.compile` methods available in `tf.keras`, then `tf.keras` will handle the
  distribution for you.
  For example:
  ```python
  my_strategy = tf.distribute.MirroredStrategy()
  with my_strategy.scope():
    @tf.function
    def distribute_train_epoch(dataset):
      def replica_fn(input):
        # process input and return result
        return result
      total_result = 0
      for x in dataset:
        per_replica_result = my_strategy.run(replica_fn, args=(x,))
        total_result += my_strategy.reduce(tf.distribute.ReduceOp.SUM,
                                           per_replica_result, axis=None)
      return total_result
    dist_dataset = my_strategy.experimental_distribute_dataset(dataset)
    for _ in range(EPOCHS):
      train_result = distribute_train_epoch(dist_dataset)
  ```
  Args:
    devices: a list of device strings such as `['/gpu:0', '/gpu:1']`.  If
      `None`, all available GPUs are used. If no GPUs are found, CPU is used.
    cross_device_ops: optional, a descendant of `CrossDeviceOps`. If this is not
      set, `NcclAllReduce()` will be used by default.  One would customize this
      if NCCL isn't available or if a special implementation that exploits
      the particular hardware is available.
  """
  # Only set this in tests.
  _collective_key_base = 0
  def __init__(self, devices=None, cross_device_ops=None):
    extended = MirroredExtended(
        self, devices=devices, cross_device_ops=cross_device_ops)
    super(MirroredStrategy, self).__init__(extended)
    distribute_lib.distribution_strategy_gauge.get_cell("V2").set(
        "MirroredStrategy")
