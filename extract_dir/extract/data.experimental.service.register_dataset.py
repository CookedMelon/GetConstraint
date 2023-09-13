@tf_export("data.experimental.service.register_dataset")
def register_dataset(service, dataset, compression="AUTO", dataset_id=None):
  """Registers a dataset with the tf.data service.
  `register_dataset` registers a dataset with the tf.data service so that
  datasets can be created later with
  `tf.data.experimental.service.from_dataset_id`. This is useful when the
  dataset
  is registered by one process, then used in another process. When the same
  process is both registering and reading from the dataset, it is simpler to use
  `tf.data.experimental.service.distribute` instead.
  If the dataset is already registered with the tf.data service,
  `register_dataset` returns the already-registered dataset's id.
  >>> dispatcher = tf.data.experimental.service.DispatchServer()
  >>> dispatcher_address = dispatcher.target.split("://")[1]
  >>> worker = tf.data.experimental.service.WorkerServer(
  ...     tf.data.experimental.service.WorkerConfig(
  ...         dispatcher_address=dispatcher_address))
  >>> dataset = tf.data.Dataset.range(10)
  >>> dataset_id = tf.data.experimental.service.register_dataset(
  ...     dispatcher.target, dataset)
  >>> dataset = tf.data.experimental.service.from_dataset_id(
  ...     processing_mode="parallel_epochs",
  ...     service=dispatcher.target,
  ...     dataset_id=dataset_id,
  ...     element_spec=dataset.element_spec)
  >>> print(list(dataset.as_numpy_iterator()))
  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  Args:
    service: A string or a tuple indicating how to connect to the tf.data
      service. If it's a string, it should be in the format
      `[<protocol>://]<address>`, where `<address>` identifies the dispatcher
        address and `<protocol>` can optionally be used to override the default
        protocol to use. If it's a tuple, it should be (protocol, address).
    dataset: A `tf.data.Dataset` to register with the tf.data service.
    compression: (Optional.) How to compress the dataset's elements before
      transferring them over the network. "AUTO" leaves the decision of how to
      compress up to the tf.data service runtime. `None` indicates not to
      compress.
    dataset_id: (Optional.) By default, tf.data service generates a unique
      (string) ID for each registered dataset. If a `dataset_id` is provided, it
      will use the specified ID. If a dataset with a matching ID already exists,
      no new dataset is registered. This is useful if multiple training jobs
      want to (re)use the same dataset for training. In this case, they can
      register the dataset with the same dataset ID.
  Returns:
    A scalar string tensor representing the dataset ID.
  """
  return _register_dataset(service, dataset, compression, dataset_id)
