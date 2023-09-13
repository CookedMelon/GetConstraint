@tf_export("distribute.Server", v1=["distribute.Server", "train.Server"])
@deprecation.deprecated_endpoints("train.Server")
class Server:
  """An in-process TensorFlow server, for use in distributed training.
  A `tf.distribute.Server` instance encapsulates a set of devices and a
  `tf.compat.v1.Session` target that
  can participate in distributed training. A server belongs to a
  cluster (specified by a `tf.train.ClusterSpec`), and
  corresponds to a particular task in a named job. The server can
  communicate with any other server in the same cluster.
  """
  def __init__(self,
               server_or_cluster_def,
               job_name=None,
               task_index=None,
               protocol=None,
               config=None,
               start=True):
    """Creates a new server with the given definition.
    The `job_name`, `task_index`, and `protocol` arguments are optional, and
    override any information provided in `server_or_cluster_def`.
    Args:
      server_or_cluster_def: A `tf.train.ServerDef` or `tf.train.ClusterDef`
        protocol buffer, or a `tf.train.ClusterSpec` object, describing the
        server to be created and/or the cluster of which it is a member.
      job_name: (Optional.) Specifies the name of the job of which the server is
        a member. Defaults to the value in `server_or_cluster_def`, if
        specified.
      task_index: (Optional.) Specifies the task index of the server in its job.
        Defaults to the value in `server_or_cluster_def`, if specified.
        Otherwise defaults to 0 if the server's job has only one task.
      protocol: (Optional.) Specifies the protocol to be used by the server.
        Acceptable values include `"grpc", "grpc+verbs"`. Defaults to the value
        in `server_or_cluster_def`, if specified. Otherwise defaults to
        `"grpc"`.
      config: (Options.) A `tf.compat.v1.ConfigProto` that specifies default
        configuration options for all sessions that run on this server.
      start: (Optional.) Boolean, indicating whether to start the server after
        creating it. Defaults to `True`.
    Raises:
      tf.errors.OpError: Or one of its subclasses if an error occurs while
        creating the TensorFlow server.
    """
    self._server_def = _make_server_def(server_or_cluster_def, job_name,
                                        task_index, protocol, config)
    self._server = c_api.TF_NewServer(self._server_def.SerializeToString())
    if start:
      self.start()
  def __del__(self):
    # At shutdown, `errors` may have been garbage collected.
    if errors is not None:
      exception = errors.UnimplementedError
    else:
      exception = Exception
    try:
      c_api.TF_ServerStop(self._server)
      # Clean shutdown of servers is not yet implemented, so
      # we leak instead of calling c_api.TF_DeleteServer here.
      # See:
      # https://github.com/tensorflow/tensorflow/blob/0495317a6e9dd4cac577b9d5cf9525e62b571018/tensorflow/core/distributed_runtime/rpc/grpc_server_lib.h#L73
    except AttributeError:
      # At shutdown, `c_api` may have been garbage collected.
      pass
    except exception:
      pass
    self._server = None
  def start(self):
    """Starts this server.
    Raises:
      tf.errors.OpError: Or one of its subclasses if an error occurs while
        starting the TensorFlow server.
    """
    c_api.TF_ServerStart(self._server)
  def join(self):
    """Blocks until the server has shut down.
    This method currently blocks forever.
    Raises:
      tf.errors.OpError: Or one of its subclasses if an error occurs while
        joining the TensorFlow server.
    """
    c_api.TF_ServerJoin(self._server)
  @property
  def server_def(self):
    """Returns the `tf.train.ServerDef` for this server.
    Returns:
      A `tf.train.ServerDef` protocol buffer that describes the configuration
      of this server.
    """
    return self._server_def
  @property
  def target(self):
    """Returns the target for a `tf.compat.v1.Session` to connect to this server.
    To create a
    `tf.compat.v1.Session` that
    connects to this server, use the following snippet:
    ```python
    server = tf.distribute.Server(...)
    with tf.compat.v1.Session(server.target):
      # ...
    ```
    Returns:
      A string containing a session target for this server.
    """
    return c_api.TF_ServerTarget(self._server)
  @staticmethod
  def create_local_server(config=None, start=True):
    """Creates a new single-process cluster running on the local host.
    This method is a convenience wrapper for creating a
    `tf.distribute.Server` with a `tf.train.ServerDef` that specifies a
    single-process cluster containing a single task in a job called
    `"local"`.
    Args:
      config: (Options.) A `tf.compat.v1.ConfigProto` that specifies default
        configuration options for all sessions that run on this server.
      start: (Optional.) Boolean, indicating whether to start the server after
        creating it. Defaults to `True`.
    Returns:
      A local `tf.distribute.Server`.
    """
    # Specifying port 0 means that the OS will choose a free port for the
    # server.
    return Server({"localhost": ["localhost:0"]},
                  protocol="grpc",
                  config=config,
                  start=start)
