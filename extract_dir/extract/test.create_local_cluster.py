@tf_export("test.create_local_cluster")
def create_local_cluster(num_workers,
                         num_ps,
                         protocol="grpc",
                         worker_config=None,
                         ps_config=None):
  """Create and start local servers and return the associated `Server` objects.
  "PS" stands for "parameter server": a task responsible for storing and
  updating the model's parameters. Other tasks send updates to these parameters
  as they work on optimizing the parameters. This particular division of labor
  between tasks is not required, but is common for distributed training.
  Read more at https://www.tensorflow.org/guide/extend/architecture
  ![components](https://www.tensorflow.org/images/diag1.svg "components")
  Figure illustrates the interaction of these components.
  "/job:worker/task:0" and "/job:ps/task:0" are both tasks with worker services.
  Example:
  ```python
  workers, _ = tf.test.create_local_cluster(num_workers=2, num_ps=2)
  worker_sessions = [tf.compat.v1.Session(w.target) for w in workers]
  with tf.device("/job:ps/task:0"):
    ...
  with tf.device("/job:ps/task:1"):
    ...
  with tf.device("/job:worker/task:0"):
    ...
  with tf.device("/job:worker/task:1"):
    ...
  worker_sessions[0].run(...)
  ```
  Args:
    num_workers: Number of worker servers to start.
    num_ps: Number of PS servers to start.
    protocol: Communication protocol. Allowed values are documented in the
      documentation of `tf.distribute.Server`.
    worker_config: (optional) `tf.ConfigProto` to initialize workers. Can be
      used to instantiate multiple devices etc.
    ps_config: (optional) `tf.ConfigProto` to initialize PS servers.
  Returns:
    A tuple `(worker_servers, ps_servers)`.  `worker_servers` is a list
    of `num_workers` objects of type `tf.distribute.Server` (all running
    locally);
    and `ps_servers` is a list of `num_ps` objects of similar type.
  Raises:
    ImportError: if portpicker module was not found at load time
  """
  worker_ports = [pick_unused_port() for _ in range(num_workers)]
  ps_ports = [pick_unused_port() for _ in range(num_ps)]
  cluster_dict = {
      "worker": ["localhost:%s" % port for port in worker_ports],
      "ps": ["localhost:%s" % port for port in ps_ports]
  }
  cs = server_lib.ClusterSpec(cluster_dict)
  workers = [
      server_lib.Server(
          cs,
          job_name="worker",
          protocol=protocol,
          task_index=ix,
          config=worker_config,
          start=True) for ix in range(num_workers)
  ]
  ps_servers = [
      server_lib.Server(
          cs,
          job_name="ps",
          protocol=protocol,
          task_index=ix,
          config=ps_config,
          start=True) for ix in range(num_ps)
  ]
  return workers, ps_servers
