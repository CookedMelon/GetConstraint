@tf_export("data.experimental.service.from_dataset_id")
def from_dataset_id(processing_mode,
                    service,
                    dataset_id,
                    element_spec=None,
                    job_name=None,
                    consumer_index=None,
                    num_consumers=None,
                    max_outstanding_requests=None,
                    data_transfer_protocol=None,
                    cross_trainer_cache=None,
                    target_workers="AUTO"):
  """Creates a dataset which reads data from the tf.data service.
  This is useful when the dataset is registered by one process, then used in
  another process. When the same process is both registering and reading from
  the dataset, it is simpler to use `tf.data.experimental.service.distribute`
  instead.
  Before using `from_dataset_id`, the dataset must have been registered with the
  tf.data service using `tf.data.experimental.service.register_dataset`.
  `register_dataset` returns a dataset id for the registered dataset. That is
  the `dataset_id` which should be passed to `from_dataset_id`.
  The `element_spec` argument indicates the `tf.TypeSpec`s for the elements
  produced by the dataset. Currently `element_spec` must be explicitly
  specified, and match the dataset registered under `dataset_id`. `element_spec`
  defaults to `None` so that in the future we can support automatically
  discovering the `element_spec` by querying the tf.data service.
  `tf.data.experimental.service.distribute` is a convenience method which
  combines `register_dataset` and `from_dataset_id` into a dataset
  transformation.
  See the documentation for `tf.data.experimental.service.distribute` for more
  detail about how `from_dataset_id` works.
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
    processing_mode: A `tf.data.experimental.service.ShardingPolicy` specifying
      how to shard the dataset among tf.data workers. See
      `tf.data.experimental.service.ShardingPolicy` for details. For backwards
      compatibility, `processing_mode` may also be set to the strings
      `"parallel_epochs"` or `"distributed_epoch"`, which are respectively
      equivalent to `ShardingPolicy.OFF` and `ShardingPolicy.DYNAMIC`.
    service: A string or a tuple indicating how to connect to the tf.data
      service. If it's a string, it should be in the format
      `[<protocol>://]<address>`, where `<address>` identifies the dispatcher
        address and `<protocol>` can optionally be used to override the default
        protocol to use. If it's a tuple, it should be (protocol, address).
    dataset_id: The id of the dataset to read from. This id is returned by
      `register_dataset` when the dataset is registered with the tf.data
      service.
    element_spec: A nested structure of `tf.TypeSpec`s representing the type of
      elements produced by the dataset. This argument is only required inside a
      tf.function. Use `tf.data.Dataset.element_spec` to get the element spec
      for a given dataset.
    job_name: (Optional.) The name of the job. If provided, it must be a
      non-empty string. This argument makes it possible for multiple datasets to
      share the same job. The default behavior is that the dataset creates
      anonymous, exclusively owned jobs.
    consumer_index: (Optional.) The index of the consumer in the range from `0`
      to `num_consumers`. Must be specified alongside `num_consumers`. When
      specified, consumers will read from the job in a strict round-robin order,
      instead of the default first-come-first-served order.
    num_consumers: (Optional.) The number of consumers which will consume from
      the job. Must be specified alongside `consumer_index`. When specified,
      consumers will read from the job in a strict round-robin order, instead of
      the default first-come-first-served order. When `num_consumers` is
      specified, the dataset must have infinite cardinality to prevent a
      producer from running out of data early and causing consumers to go out of
      sync.
    max_outstanding_requests: (Optional.) A limit on how many elements may be
      requested at the same time. You can use this option to control the amount
      of memory used, since `distribute` won't use more than `element_size` *
      `max_outstanding_requests` of memory.
    data_transfer_protocol: (Optional.) The protocol to use for transferring
      data with the tf.data service. By default, data is transferred using gRPC.
    cross_trainer_cache: (Optional.) If a `CrossTrainerCache` object is
      provided, dataset iteration will be shared across concurrently running
      trainers. See
      https://www.tensorflow.org/api_docs/python/tf/data/experimental/service#sharing_tfdata_service_with_concurrent_trainers
      for details.
    target_workers: (Optional.) Which workers to read from. If `"AUTO"`, tf.data
      runtime decides which workers to read from. If `"ANY"`, reads from any
      tf.data service workers. If `"LOCAL"`, only reads from local in-processs
      tf.data service workers. `"AUTO"` works well for most cases, while users
      can specify other targets. For example, `"LOCAL"` helps avoid RPCs and
      data copy if every TF worker colocates with a tf.data service worker.
      Consumers of a shared job must use the same `target_workers`. Defaults to
      `"AUTO"`.
  Returns:
    A `tf.data.Dataset` which reads from the tf.data service.
  """
  _validate_job_name(job_name)
  if job_name is not None:
    job_name = string_ops.string_join(
        ["dataset_id=", _to_string(dataset_id), job_name], "/")
  return _from_dataset_id(
      processing_mode=processing_mode,
      service=service,
      dataset_id=dataset_id,
      element_spec=element_spec,
      job_name=job_name,
      consumer_index=consumer_index,
      num_consumers=num_consumers,
      max_outstanding_requests=max_outstanding_requests,
      data_transfer_protocol=data_transfer_protocol,
      cross_trainer_cache=cross_trainer_cache,
      target_workers=target_workers)
