@tf_export("train.ClusterSpec")
class ClusterSpec:
  """Represents a cluster as a set of "tasks", organized into "jobs".
  A `tf.train.ClusterSpec` represents the set of processes that
  participate in a distributed TensorFlow computation. Every
  `tf.distribute.Server` is constructed in a particular cluster.
  To create a cluster with two jobs and five tasks, you specify the
  mapping from job names to lists of network addresses (typically
  hostname-port pairs).
  ```python
  cluster = tf.train.ClusterSpec({"worker": ["worker0.example.com:2222",
                                             "worker1.example.com:2222",
                                             "worker2.example.com:2222"],
                                  "ps": ["ps0.example.com:2222",
                                         "ps1.example.com:2222"]})
  ```
  Each job may also be specified as a sparse mapping from task indices
  to network addresses. This enables a server to be configured without
  needing to know the identity of (for example) all other worker
  tasks:
  ```python
  cluster = tf.train.ClusterSpec({"worker": {1: "worker1.example.com:2222"},
                                  "ps": ["ps0.example.com:2222",
                                         "ps1.example.com:2222"]})
  ```
  """
  def __init__(self, cluster):
    """Creates a `ClusterSpec`.
    Args:
      cluster: A dictionary mapping one or more job names to (i) a list of
        network addresses, or (ii) a dictionary mapping integer task indices to
        network addresses; or a `tf.train.ClusterDef` protocol buffer.
    Raises:
      TypeError: If `cluster` is not a dictionary mapping strings to lists
        of strings, and not a `tf.train.ClusterDef` protobuf.
    """
    if isinstance(cluster, dict):
      self._cluster_spec = {}
      for job_name, tasks in cluster.items():
        if isinstance(tasks, (list, tuple)):
          job_tasks = {i: task for i, task in enumerate(tasks)}
        elif isinstance(tasks, dict):
          job_tasks = {int(i): task for i, task in tasks.items()}
        else:
          raise TypeError("The tasks for job %r must be a list or a dictionary "
                          "from integers to strings." % job_name)
        self._cluster_spec[job_name] = job_tasks
      self._make_cluster_def()
    elif isinstance(cluster, cluster_pb2.ClusterDef):
      self._cluster_def = cluster
      self._cluster_spec = {}
      for job_def in self._cluster_def.job:
        self._cluster_spec[job_def.name] = {
            i: t for i, t in job_def.tasks.items()
        }
    elif isinstance(cluster, ClusterSpec):
      self._cluster_def = cluster_pb2.ClusterDef()
      self._cluster_def.MergeFrom(cluster.as_cluster_def())
      self._cluster_spec = {}
      for job_def in self._cluster_def.job:
        self._cluster_spec[job_def.name] = {
            i: t for i, t in job_def.tasks.items()
        }
    else:
      raise TypeError("`cluster` must be a dictionary mapping one or more "
                      "job names to lists of network addresses, or a "
                      "`ClusterDef` protocol buffer")
  def __bool__(self):
    return bool(self._cluster_spec)
  # Python 2.x
  __nonzero__ = __bool__
  def __eq__(self, other):
    return self._cluster_spec == other
  def __ne__(self, other):
    return self._cluster_spec != other
  def __repr__(self):
    key_values = self.as_dict()
    string_items = [
        repr(k) + ": " + repr(key_values[k]) for k in sorted(key_values)
    ]
    return "ClusterSpec({" + ", ".join(string_items) + "})"
  def as_dict(self):
    """Returns a dictionary from job names to their tasks.
    For each job, if the task index space is dense, the corresponding
    value will be a list of network addresses; otherwise it will be a
    dictionary mapping (sparse) task indices to the corresponding
    addresses.
    Returns:
      A dictionary mapping job names to lists or dictionaries
      describing the tasks in those jobs.
    """
    ret = {}
    for job in self.jobs:
      task_indices = self.task_indices(job)
      if len(task_indices) == 0:
        ret[job] = {}
        continue
      if max(task_indices) + 1 == len(task_indices):
        # Return a list because the task indices are dense. This
        # matches the behavior of `as_dict()` before support for
        # sparse jobs was added.
        ret[job] = self.job_tasks(job)
      else:
        ret[job] = {i: self.task_address(job, i) for i in task_indices}
    return ret
  def as_cluster_def(self):
    """Returns a `tf.train.ClusterDef` protocol buffer based on this cluster."""
    return self._cluster_def
  @property
  def jobs(self):
    """Returns a list of job names in this cluster.
    Returns:
      A list of strings, corresponding to the names of jobs in this cluster.
    """
    return list(self._cluster_spec.keys())
  def num_tasks(self, job_name):
    """Returns the number of tasks defined in the given job.
    Args:
      job_name: The string name of a job in this cluster.
    Returns:
      The number of tasks defined in the given job.
    Raises:
      ValueError: If `job_name` does not name a job in this cluster.
    """
    try:
      job = self._cluster_spec[job_name]
    except KeyError:
      raise ValueError("No such job in cluster: %r" % job_name)
    return len(job)
  def task_indices(self, job_name):
    """Returns a list of valid task indices in the given job.
    Args:
      job_name: The string name of a job in this cluster.
    Returns:
      A list of valid task indices in the given job.
    Raises:
      ValueError: If `job_name` does not name a job in this cluster,
      or no task with index `task_index` is defined in that job.
    """
    try:
      job = self._cluster_spec[job_name]
    except KeyError:
      raise ValueError("No such job in cluster: %r" % job_name)
    return list(sorted(job.keys()))
  def task_address(self, job_name, task_index):
    """Returns the address of the given task in the given job.
    Args:
      job_name: The string name of a job in this cluster.
      task_index: A non-negative integer.
    Returns:
      The address of the given task in the given job.
    Raises:
      ValueError: If `job_name` does not name a job in this cluster,
      or no task with index `task_index` is defined in that job.
    """
    try:
      job = self._cluster_spec[job_name]
    except KeyError:
      raise ValueError("No such job in cluster: %r" % job_name)
    try:
      return job[task_index]
    except KeyError:
      raise ValueError("No task with index %r in job %r" %
                       (task_index, job_name))
  def job_tasks(self, job_name):
    """Returns a mapping from task ID to address in the given job.
    NOTE: For backwards compatibility, this method returns a list. If
    the given job was defined with a sparse set of task indices, the
    length of this list may not reflect the number of tasks defined in
    this job. Use the `tf.train.ClusterSpec.num_tasks` method
    to find the number of tasks defined in a particular job.
    Args:
      job_name: The string name of a job in this cluster.
    Returns:
      A list of task addresses, where the index in the list
      corresponds to the task index of each task. The list may contain
      `None` if the job was defined with a sparse set of task indices.
    Raises:
      ValueError: If `job_name` does not name a job in this cluster.
    """
    try:
      job = self._cluster_spec[job_name]
    except KeyError:
      raise ValueError("No such job in cluster: %r" % job_name)
    ret = [None for _ in range(max(job.keys()) + 1)]
    for i, task in job.items():
      ret[i] = task
    return ret
  def _make_cluster_def(self):
    """Creates a `tf.train.ClusterDef` based on the given `cluster_spec`.
    Raises:
      TypeError: If `cluster_spec` is not a dictionary mapping strings to lists
        of strings.
    """
    self._cluster_def = cluster_pb2.ClusterDef()
    # NOTE(mrry): Sort by job_name to produce deterministic protobufs.
    for job_name, tasks in sorted(self._cluster_spec.items()):
      try:
        job_name = compat.as_bytes(job_name)
      except TypeError:
        raise TypeError("Job name %r must be bytes or unicode" % job_name)
      job_def = self._cluster_def.job.add()
      job_def.name = job_name
      for i, task_address in sorted(tasks.items()):
        try:
          task_address = compat.as_bytes(task_address)
        except TypeError:
          raise TypeError("Task address %r must be bytes or unicode" %
                          task_address)
        job_def.tasks[i] = task_address
