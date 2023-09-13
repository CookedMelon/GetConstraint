@tf_export("distribute.MultiWorkerMirroredStrategy", v1=[])
class CollectiveAllReduceStrategy(distribute_lib.Strategy):
  """A distribution strategy for synchronous training on multiple workers.
  This strategy implements synchronous distributed training across multiple
  workers, each with potentially multiple GPUs. Similar to
  `tf.distribute.MirroredStrategy`, it replicates all variables and computations
  to each local device. The difference is that it uses a distributed collective
  implementation (e.g. all-reduce), so that multiple workers can work together.
  You need to launch your program on each worker and configure
  `cluster_resolver` correctly. For example, if you are using
  `tf.distribute.cluster_resolver.TFConfigClusterResolver`, each worker needs to
  have its corresponding `task_type` and `task_id` set in the `TF_CONFIG`
  environment variable. An example TF_CONFIG on worker-0 of a two worker cluster
  is:
  ```
  TF_CONFIG = '{"cluster": {"worker": ["localhost:12345", "localhost:23456"]}, "task": {"type": "worker", "index": 0} }'
  ```
  Your program runs on each worker as-is. Note that collectives require each
  worker to participate. All `tf.distribute` and non `tf.distribute` API may use
  collectives internally, e.g. checkpointing and saving since reading a
  `tf.Variable` with `tf.VariableSynchronization.ON_READ` all-reduces the value.
  Therefore it's recommended to run exactly the same program on each worker.
  Dispatching based on `task_type` or `task_id` of the worker is error-prone.
  `cluster_resolver.num_accelerators()` determines the number of GPUs the
  strategy uses. If it's zero, the strategy uses the CPU. All workers need to
  use the same number of devices, otherwise the behavior is undefined.
  This strategy is not intended for TPU. Use `tf.distribute.TPUStrategy`
  instead.
  After setting up TF_CONFIG, using this strategy is similar to using
  `tf.distribute.MirroredStrategy` and `tf.distribute.TPUStrategy`.
  ```
  strategy = tf.distribute.MultiWorkerMirroredStrategy()
  with strategy.scope():
    model = tf.keras.Sequential([
      tf.keras.layers.Dense(2, input_shape=(5,)),
    ])
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
  def dataset_fn(ctx):
    x = np.random.random((2, 5)).astype(np.float32)
    y = np.random.randint(2, size=(2, 1))
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    return dataset.repeat().batch(1, drop_remainder=True)
  dist_dataset = strategy.distribute_datasets_from_function(dataset_fn)
  model.compile()
  model.fit(dist_dataset)
  ```
  You can also write your own training loop:
  ```
  @tf.function
  def train_step(iterator):
    def step_fn(inputs):
      features, labels = inputs
      with tf.GradientTape() as tape:
        logits = model(features, training=True)
        loss = tf.keras.losses.sparse_categorical_crossentropy(
            labels, logits)
      grads = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(grads, model.trainable_variables))
    strategy.run(step_fn, args=(next(iterator),))
  for _ in range(NUM_STEP):
    train_step(iterator)
  ```
  See
  [Multi-worker training with Keras](https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras)
  for a detailed tutorial.
  __Saving__
  You need to save and checkpoint on all workers instead of just one. This is
  because variables whose synchronization=ON_READ triggers aggregation during
  saving. It's recommended to save to a different path on each worker to avoid
  race conditions. Each worker saves the same thing. See
  [Multi-worker training with Keras](https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras#model_saving_and_loading)
  tutorial for examples.
  __Known Issues__
  * `tf.distribute.cluster_resolver.TFConfigClusterResolver` does not return the
  correct number of accelerators. The strategy uses all available GPUs if
  `cluster_resolver` is `tf.distribute.cluster_resolver.TFConfigClusterResolver`
  or `None`.
  * In eager mode, the strategy needs to be created before calling any other
  Tensorflow API.
  """
  # pylint: enable=line-too-long
  # TODO(anjalisridhar): Update our guides with examples showing how we can use
  # the cluster_resolver argument.
  # The starting number for collective keys. This should only be set in tests.
  _collective_key_base = 0
  def __init__(self,
               cluster_resolver=None,
               communication_options=None):
    """Creates the strategy.
    Args:
      cluster_resolver: optional
        `tf.distribute.cluster_resolver.ClusterResolver`. If `None`,
        `tf.distribute.cluster_resolver.TFConfigClusterResolver` is used.
      communication_options: optional
        `tf.distribute.experimental.CommunicationOptions`. This configures the
        default options for cross device communications. It can be overridden by
        options provided to the communication APIs like
        `tf.distribute.ReplicaContext.all_reduce`. See
        `tf.distribute.experimental.CommunicationOptions` for details.
    """
    if communication_options is None:
      communication_options = collective_util.Options()
    super(CollectiveAllReduceStrategy, self).__init__(
        CollectiveAllReduceExtended(
            self,
            cluster_resolver=cluster_resolver,
            communication_options=communication_options))
    distribute_lib.distribution_strategy_gauge.get_cell("V2").set(
        "MultiWorkerMirroredStrategy")
    # pylint: disable=protected-access
    distribute_lib.distribution_strategy_replica_gauge.get_cell(
        "num_workers").set(self.extended._num_workers)
    distribute_lib.distribution_strategy_replica_gauge.get_cell(
        "num_replicas_per_worker").set(self.extended._num_devices_per_worker)
  @classmethod
  def _from_local_devices(cls, devices, communication_options=None):
    """A convenience method to create an object with a list of devices."""
    obj = cls(communication_options=communication_options)
    obj.extended._initialize_local(TFConfigClusterResolver(), devices=devices)  # pylint: disable=protected-access
    return obj
  @property
  def cluster_resolver(self):
    """Returns the cluster resolver associated with this strategy.
    As a multi-worker strategy, `tf.distribute.MultiWorkerMirroredStrategy`
    provides the associated `tf.distribute.cluster_resolver.ClusterResolver`. If
    the user provides one in `__init__`, that instance is returned; if the user
    does not, a default `TFConfigClusterResolver` is provided.
    """
    return self.extended._cluster_resolver  # pylint: disable=protected-access
