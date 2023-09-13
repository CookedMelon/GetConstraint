@tf_export("config.experimental_connect_to_host")
def connect_to_remote_host(remote_host=None, job_name="worker"):
  """Connects to a single machine to enable remote execution on it.
  Will make devices on the remote host available to use. Note that calling this
  more than once will work, but will invalidate any tensor handles on the old
  remote devices.
  Using the default job_name of worker, you can schedule ops to run remotely as
  follows:
  ```python
  # When eager execution is enabled, connect to the remote host.
  tf.config.experimental_connect_to_host("exampleaddr.com:9876")
  with ops.device("job:worker/replica:0/task:1/device:CPU:0"):
    # The following tensors should be resident on the remote device, and the op
    # will also execute remotely.
    x1 = array_ops.ones([2, 2])
    x2 = array_ops.ones([2, 2])
    y = math_ops.matmul(x1, x2)
  ```
  Args:
    remote_host: a single or a list the remote server addr in host-port format.
    job_name: The job name under which the new server will be accessible.
  Raises:
    ValueError: if remote_host is None.
  """
  if not remote_host:
    raise ValueError("Must provide at least one remote_host")
  remote_hosts = nest.flatten(remote_host)
  cluster_spec = server_lib.ClusterSpec(
      {job_name: [_strip_prefix(host, _GRPC_PREFIX) for host in remote_hosts]})
  connect_to_cluster(cluster_spec)
