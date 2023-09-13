@tf_export("data.experimental.service.DispatchServer", v1=[])
class DispatchServer:
  """An in-process tf.data service dispatch server.
  A `tf.data.experimental.service.DispatchServer` coordinates a cluster of
  `tf.data.experimental.service.WorkerServer`s. When the workers start, they
  register themselves with the dispatcher.
  >>> dispatcher = tf.data.experimental.service.DispatchServer()
  >>> dispatcher_address = dispatcher.target.split("://")[1]
  >>> worker = tf.data.experimental.service.WorkerServer(
  ...     tf.data.experimental.service.WorkerConfig(
  ...     dispatcher_address=dispatcher_address))
  >>> dataset = tf.data.Dataset.range(10)
  >>> dataset = dataset.apply(tf.data.experimental.service.distribute(
  ...     processing_mode="parallel_epochs", service=dispatcher.target))
  >>> print(list(dataset.as_numpy_iterator()))
  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  When starting a dedicated tf.data dispatch process, use join() to block
  after starting up the server, until the server terminates.
  ```
  dispatcher = tf.data.experimental.service.DispatchServer(
      tf.data.experimental.service.DispatcherConfig(port=5050))
  dispatcher.join()
  ```
  Call stop() to gracefully terminate the dispatcher. The server automatically
  stops when all reference to it have been deleted.
  To start a `DispatchServer` in fault-tolerant mode, set `work_dir` and
  `fault_tolerant_mode` like below:
  ```
  dispatcher = tf.data.experimental.service.DispatchServer(
      tf.data.experimental.service.DispatcherConfig(
          port=5050,
          work_dir="gs://my-bucket/dispatcher/work_dir",
          fault_tolerant_mode=True))
  ```
  """
  def __init__(self, config=None, start=True):
    """Creates a new dispatch server.
    Args:
      config: (Optional.) A `tf.data.experimental.service.DispatcherConfig`
        configration. If `None`, the dispatcher will use default
        configuration values.
      start: (Optional.) Boolean, indicating whether to start the server after
        creating it. Defaults to True.
    """
    config = config or DispatcherConfig()
    if config.fault_tolerant_mode and not config.work_dir:
      raise ValueError(
          "Cannot enable fault tolerant mode without configuring a work dir. "
          "Make sure to set `work_dir` in the `config` object passed to "
          "`DispatcherServer`.")
    self._config = config
    if isinstance(config, service_config_pb2.DispatcherConfig):
      config_proto = config
    else:
      config_proto = service_config_pb2.DispatcherConfig(
          port=config.port,
          protocol=config.protocol,
          work_dir=config.work_dir,
          fault_tolerant_mode=config.fault_tolerant_mode,
          worker_addresses=config.worker_addresses,
          job_gc_check_interval_ms=config.job_gc_check_interval_ms,
          job_gc_timeout_ms=config.job_gc_timeout_ms,
          worker_timeout_ms=config.worker_timeout_ms,
      )
    self._server = _pywrap_server_lib.TF_DATA_NewDispatchServer(
        config_proto.SerializeToString())
    if start:
      self._server.start()
  def start(self):
    """Starts this server.
    >>> dispatcher = tf.data.experimental.service.DispatchServer(start=False)
    >>> dispatcher.start()
    Raises:
      tf.errors.OpError: Or one of its subclasses if an error occurs while
        starting the server.
    """
    self._server.start()
  def join(self):
    """Blocks until the server has shut down.
    This is useful when starting a dedicated dispatch process.
    ```
    dispatcher = tf.data.experimental.service.DispatchServer(
        tf.data.experimental.service.DispatcherConfig(port=5050))
    dispatcher.join()
    ```
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
  @property
  def target(self):
    """Returns a target that can be used to connect to the server.
    >>> dispatcher = tf.data.experimental.service.DispatchServer()
    >>> dataset = tf.data.Dataset.range(10)
    >>> dataset = dataset.apply(tf.data.experimental.service.distribute(
    ...     processing_mode="parallel_epochs", service=dispatcher.target))
    The returned string will be in the form protocol://address, e.g.
    "grpc://localhost:5050".
    """
    return "{0}://localhost:{1}".format(self._config.protocol,
                                        self._server.bound_port())
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
  def _num_workers(self):
    """Returns the number of workers registered with the dispatcher."""
    return self._server.num_workers()
  def _snapshot_streams(self, path):
    """Returns information about all the streams for a snapshot."""
    return self._server.snapshot_streams(path)
