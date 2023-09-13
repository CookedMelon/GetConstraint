@tf_export("data.experimental.service.WorkerConfig")
class WorkerConfig(
    collections.namedtuple("WorkerConfig", [
        "dispatcher_address", "worker_address", "port", "protocol",
        "heartbeat_interval_ms", "dispatcher_timeout_ms",
        "data_transfer_protocol", "data_transfer_address"
    ])):
  """Configuration class for tf.data service dispatchers.
  Fields:
    dispatcher_address: Specifies the address of the dispatcher.
    worker_address: Specifies the address of the worker server. This address is
      passed to the dispatcher so that the dispatcher can tell clients how to
      connect to this worker.
    port: Specifies the port to bind to. A value of 0 indicates that the worker
      can bind to any available port.
    protocol: A string indicating the protocol to be used by the worker to
      connect to the dispatcher. E.g. "grpc".
    heartbeat_interval_ms: How often the worker should heartbeat to the
      dispatcher, in milliseconds. If not set, the runtime will select a
      reasonable default. A higher value will reduce the load on the dispatcher,
      while a lower value will reduce the time it takes to reclaim resources
      from finished jobs.
    dispatcher_timeout_ms: How long, in milliseconds, to retry requests to the
      dispatcher before giving up and reporting an error. Defaults to 1 hour.
    data_transfer_protocol: A string indicating the protocol to be used by the
      worker to transfer data to the client. E.g. "grpc".
    data_transfer_address: A string indicating the data transfer address of the
      worker server.
  """
  def __new__(cls,
              dispatcher_address,
              worker_address=None,
              port=0,
              protocol=None,
              heartbeat_interval_ms=None,
              dispatcher_timeout_ms=None,
              data_transfer_protocol=None,
              data_transfer_address=None):
    if worker_address is None:
      worker_address = "localhost:%port%"
    if protocol is None:
      protocol = _pywrap_utils.TF_DATA_DefaultProtocol()
    if data_transfer_address is None:
      data_transfer_address = "localhost:%port%"
    heartbeat_interval_ms = _get_time_or_placeholder(heartbeat_interval_ms)
    dispatcher_timeout_ms = _get_time_or_placeholder(dispatcher_timeout_ms)
    return super(WorkerConfig,
                 cls).__new__(cls, dispatcher_address, worker_address, port,
                              protocol, heartbeat_interval_ms,
                              dispatcher_timeout_ms, data_transfer_protocol,
                              data_transfer_address)
