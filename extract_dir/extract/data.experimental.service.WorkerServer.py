@tf_export("data.experimental.service.WorkerServer", v1=[])
class WorkerServer:
  """An in-process tf.data service worker server.
  A `tf.data.experimental.service.WorkerServer` performs `tf.data.Dataset`
  processing for user-defined datasets, and provides the resulting elements over
  RPC. A worker is associated with a single
  `tf.data.experimental.service.DispatchServer`.
  >>> dispatcher = tf.data.experimental.service.DispatchServer()
  >>> dispatcher_address = dispatcher.target.split("://")[1]
  >>> worker = tf.data.experimental.service.WorkerServer(
  ...     tf.data.experimental.service.WorkerConfig(
  ...         dispatcher_address=dispatcher_address))
  >>> dataset = tf.data.Dataset.range(10)
  >>> dataset = dataset.apply(tf.data.experimental.service.distribute(
  ...     processing_mode="parallel_epochs", service=dispatcher.target))
  >>> print(list(dataset.as_numpy_iterator()))
  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  When starting a dedicated tf.data worker process, use join() to block
  after starting up the worker, until the worker terminates.
  ```
  worker = tf.data.experimental.service.WorkerServer(
      port=5051, dispatcher_address="localhost:5050")
  worker.join()
  ```
  Call stop() to gracefully terminate the worker. The worker automatically stops
  when all reference to it have been deleted.
  """
  def __init__(self, config, start=True):
    """Creates a new worker server.
    Args:
      config: A `tf.data.experimental.service.WorkerConfig` configration.
      start: (Optional.) Boolean, indicating whether to start the server after
        creating it. Defaults to True.
    """
    if config.dispatcher_address is None:
      raise ValueError(
          "Must specify a `dispatcher_address` in the `config` passed "
          "to `WorkerServer`.")
    if isinstance(config, service_config_pb2.WorkerConfig):
      config_proto = config
    else:
      config_proto = service_config_pb2.WorkerConfig(
          dispatcher_address=config.dispatcher_address,
          worker_address=config.worker_address,
          port=config.port,
          protocol=config.protocol,
          heartbeat_interval_ms=config.heartbeat_interval_ms,
          dispatcher_timeout_ms=config.dispatcher_timeout_ms,
          data_transfer_protocol=config.data_transfer_protocol,
          data_transfer_address=config.data_transfer_address)
    self._server = _pywrap_server_lib.TF_DATA_NewWorkerServer(
        config_proto.SerializeToString())
    if start:
      self._server.start()
  def start(self):
    """Starts this server.
    Raises:
      tf.errors.OpError: Or one of its subclasses if an error occurs while
        starting the server.
    """
    self._server.start()
  def join(self):
    """Blocks until the server has shut down.
    This is useful when starting a dedicated worker process.
    ```
    worker_server = tf.data.experimental.service.WorkerServer(
        port=5051, dispatcher_address="localhost:5050")
    worker_server.join()
    ```
    This method currently blocks forever.
    Raises:
      tf.errors.OpError: Or one of its subclasses if an error occurs while
        joining the server.
    """
    self._server.join()
  def stop(self):
    """Stops the server.
    Raises:
      tf.errors.OpError: Or one of its subclasses if an error occurs while
        stopping the server.
    """
    self._stop()
  def _stop(self):
    """Stops the server.
    Raises:
      tf.errors.OpError: Or one of its subclasses if an error occurs while
        stopping the server.
    """
    self._server.stop()
  def __del__(self):
    self._stop()
  @property
  def _address(self):
    """Returns the address of the server.
    The returned string will be in the form address:port, e.g. "localhost:1000".
    """
    return "localhost:{0}".format(self._server.bound_port())
  def _num_tasks(self):
    """Returns the number of tasks currently being executed on the worker."""
    return self._server.num_tasks()
  def _snapshot_task_progresses(self):
    """Returns the progresses of the snapshot tasks currently being executed.
    Returns:
      An `Iterable[common_pb2.SnapshotTaskProgress]`.
    """
    return self._server.snapshot_task_progresses()
