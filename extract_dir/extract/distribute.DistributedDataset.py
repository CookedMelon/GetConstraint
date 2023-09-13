@tf_export("distribute.DistributedDataset", v1=[])
class DistributedDatasetInterface(Iterable):
  # pylint: disable=line-too-long
  """Represents a dataset distributed among devices and machines.
  A `tf.distribute.DistributedDataset` could be thought of as a "distributed"
  dataset. When you use `tf.distribute` API to scale training to multiple
  devices or machines, you also need to distribute the input data, which leads
  to a `tf.distribute.DistributedDataset` instance, instead of a
  `tf.data.Dataset` instance in the non-distributed case. In TF 2.x,
  `tf.distribute.DistributedDataset` objects are Python iterables.
  Note: `tf.distribute.DistributedDataset` instances are *not* of type
  `tf.data.Dataset`. It only supports two usages we will mention below:
  iteration and `element_spec`. We don't support any other APIs to transform or
  inspect the dataset.
  There are two APIs to create a `tf.distribute.DistributedDataset` object:
  `tf.distribute.Strategy.experimental_distribute_dataset(dataset)`and
  `tf.distribute.Strategy.distribute_datasets_from_function(dataset_fn)`.
  *When to use which?* When you have a `tf.data.Dataset` instance, and the
  regular batch splitting (i.e. re-batch the input `tf.data.Dataset` instance
  with a new batch size that is equal to the global batch size divided by the
  number of replicas in sync) and autosharding (i.e. the
  `tf.data.experimental.AutoShardPolicy` options) work for you, use the former
  API. Otherwise, if you are *not* using a canonical `tf.data.Dataset` instance,
  or you would like to customize the batch splitting or sharding, you can wrap
  these logic in a `dataset_fn` and use the latter API. Both API handles
  prefetch to device for the user. For more details and examples, follow the
  links to the APIs.
  There are two main usages of a `DistributedDataset` object:
  1. Iterate over it to generate the input for a single device or multiple
  devices, which is a `tf.distribute.DistributedValues` instance. To do this,
  you can:
    * use a pythonic for-loop construct:
      >>> global_batch_size = 4
      >>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
      >>> dataset = tf.data.Dataset.from_tensors(([1.],[1.])).repeat(4).batch(global_batch_size)
      >>> dist_dataset = strategy.experimental_distribute_dataset(dataset)
      >>> @tf.function
      ... def train_step(input):
      ...   features, labels = input
      ...   return labels - 0.3 * features
      >>> for x in dist_dataset:
      ...   # train_step trains the model using the dataset elements
      ...   loss = strategy.run(train_step, args=(x,))
      ...   print("Loss is", loss)
      Loss is PerReplica:{
        0: tf.Tensor(
      [[0.7]
       [0.7]], shape=(2, 1), dtype=float32),
        1: tf.Tensor(
      [[0.7]
       [0.7]], shape=(2, 1), dtype=float32)
      }
      Placing the loop inside a `tf.function` will give a performance boost.
      However `break` and `return` are currently not supported if the loop is
      placed inside a `tf.function`. We also don't support placing the loop
      inside a `tf.function` when using
      `tf.distribute.experimental.MultiWorkerMirroredStrategy` or
      `tf.distribute.experimental.TPUStrategy` with multiple workers.
    * use `__iter__` to create an explicit iterator, which is of type
      `tf.distribute.DistributedIterator`
      >>> global_batch_size = 4
      >>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
      >>> train_dataset = tf.data.Dataset.from_tensors(([1.],[1.])).repeat(50).batch(global_batch_size)
      >>> train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
      >>> @tf.function
      ... def distributed_train_step(dataset_inputs):
      ...   def train_step(input):
      ...     loss = tf.constant(0.1)
      ...     return loss
      ...   per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
      ...   return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,axis=None)
      >>> EPOCHS = 2
      >>> STEPS = 3
      >>> for epoch in range(EPOCHS):
      ...   total_loss = 0.0
      ...   num_batches = 0
      ...   dist_dataset_iterator = iter(train_dist_dataset)
      ...   for _ in range(STEPS):
      ...     total_loss += distributed_train_step(next(dist_dataset_iterator))
      ...     num_batches += 1
      ...   average_train_loss = total_loss / num_batches
      ...   template = ("Epoch {}, Loss: {:.4f}")
      ...   print (template.format(epoch+1, average_train_loss))
      Epoch 1, Loss: 0.2000
      Epoch 2, Loss: 0.2000
    To achieve a performance improvement, you can also wrap the `strategy.run`
    call with a `tf.range` inside a `tf.function`. This runs multiple steps in a
    `tf.function`. Autograph will convert it to a `tf.while_loop` on the worker.
    However, it is less flexible comparing with running a single step inside
    `tf.function`. For example, you cannot run things eagerly or arbitrary
    python code within the steps.
  2. Inspect the `tf.TypeSpec` of the data generated by `DistributedDataset`.
    `tf.distribute.DistributedDataset` generates
    `tf.distribute.DistributedValues` as input to the devices. If you pass the
    input to a `tf.function` and would like to specify the shape and type of
    each Tensor argument to the function, you can pass a `tf.TypeSpec` object to
    the `input_signature` argument of the `tf.function`. To get the
    `tf.TypeSpec` of the input, you can use the `element_spec` property of the
    `tf.distribute.DistributedDataset` or `tf.distribute.DistributedIterator`
    object.
    For example:
    >>> global_batch_size = 4
    >>> epochs = 1
    >>> steps_per_epoch = 1
    >>> mirrored_strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
    >>> dataset = tf.data.Dataset.from_tensors(([2.])).repeat(100).batch(global_batch_size)
    >>> dist_dataset = mirrored_strategy.experimental_distribute_dataset(dataset)
    >>> @tf.function(input_signature=[dist_dataset.element_spec])
    ... def train_step(per_replica_inputs):
    ...   def step_fn(inputs):
    ...     return tf.square(inputs)
    ...   return mirrored_strategy.run(step_fn, args=(per_replica_inputs,))
    >>> for _ in range(epochs):
    ...   iterator = iter(dist_dataset)
    ...   for _ in range(steps_per_epoch):
    ...     output = train_step(next(iterator))
    ...     print(output)
    PerReplica:{
      0: tf.Tensor(
    [[4.]
     [4.]], shape=(2, 1), dtype=float32),
      1: tf.Tensor(
    [[4.]
     [4.]], shape=(2, 1), dtype=float32)
    }
  Visit the [tutorial](https://www.tensorflow.org/tutorials/distribute/input)
  on distributed input for more examples and caveats.
  """
  def __iter__(self):
    """Creates an iterator for the `tf.distribute.DistributedDataset`.
    The returned iterator implements the Python Iterator protocol.
    Example usage:
    >>> global_batch_size = 4
    >>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
    >>> dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4]).repeat().batch(global_batch_size)
    >>> distributed_iterator = iter(strategy.experimental_distribute_dataset(dataset))
    >>> print(next(distributed_iterator))
    PerReplica:{
      0: tf.Tensor([1 2], shape=(2,), dtype=int32),
      1: tf.Tensor([3 4], shape=(2,), dtype=int32)
    }
    Returns:
      An `tf.distribute.DistributedIterator` instance for the given
      `tf.distribute.DistributedDataset` object to enumerate over the
      distributed data.
    """
    raise NotImplementedError("Must be implemented in descendants")
  @property
  def element_spec(self):
    """The type specification of an element of this `tf.distribute.DistributedDataset`.
    Example usage:
    >>> global_batch_size = 16
    >>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
    >>> dataset = tf.data.Dataset.from_tensors(([1.],[2])).repeat(100).batch(global_batch_size)
    >>> dist_dataset = strategy.experimental_distribute_dataset(dataset)
    >>> dist_dataset.element_spec
    (PerReplicaSpec(TensorSpec(shape=(None, 1), dtype=tf.float32, name=None),
                    TensorSpec(shape=(None, 1), dtype=tf.float32, name=None)),
     PerReplicaSpec(TensorSpec(shape=(None, 1), dtype=tf.int32, name=None),
                    TensorSpec(shape=(None, 1), dtype=tf.int32, name=None)))
    Returns:
      A nested structure of `tf.TypeSpec` objects matching the structure of an
      element of this `tf.distribute.DistributedDataset`. This returned value is
      typically a `tf.distribute.DistributedValues` object and specifies the
      `tf.TensorSpec` of individual components.
    """
    raise NotImplementedError(
        "DistributedDataset.element_spec must be implemented in descendants.")
  @doc_controls.do_not_generate_docs
  def reduce(self, initial_state, reduce_func):
    raise NotImplementedError(
        "DistributedDataset.reduce must be implemented in descendants.")
