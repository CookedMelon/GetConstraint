@tf_export("config.experimental.ClusterDeviceFilters")
class ClusterDeviceFilters:
  """Represent a collection of device filters for the remote workers in cluster.
  NOTE: this is an experimental API and subject to changes.
  Set device filters for selective jobs and tasks. For each remote worker, the
  device filters are a list of strings. When any filters are present, the remote
  worker will ignore all devices which do not match any of its filters. Each
  filter can be partially specified, e.g. "/job:ps", "/job:worker/replica:3",
  etc. Note that a device is always visible to the worker it is located on.
  For example, to set the device filters for a parameter server cluster:
  ```python
  cdf = tf.config.experimental.ClusterDeviceFilters()
  for i in range(num_workers):
    cdf.set_device_filters('worker', i, ['/job:ps'])
  for i in range(num_ps):
    cdf.set_device_filters('ps', i, ['/job:worker'])
  tf.config.experimental_connect_to_cluster(cluster_def,
                                            cluster_device_filters=cdf)
  ```
  The device filters can be partically specified. For remote tasks that do not
  have device filters specified, all devices will be visible to them.
  """
  def __init__(self):
    # `_device_filters` is a dict mapping job names to job device filters.
    # Job device filters further maps task IDs to task device filters.
    # Task device filters are a list of strings, each one is a device filter.
    self._device_filters = {}
    # Serialized protobuf for cluster device filters.
    self._cluster_device_filters = None
  def set_device_filters(self, job_name, task_index, device_filters):
    """Set the device filters for given job name and task id."""
    assert all(isinstance(df, str) for df in device_filters)
    self._device_filters.setdefault(job_name, {})
    self._device_filters[job_name][task_index] = [df for df in device_filters]
    # Due to updates in data, invalidate the serialized proto cache.
    self._cluster_device_filters = None
  def _as_cluster_device_filters(self):
    """Returns a serialized protobuf of cluster device filters."""
    if self._cluster_device_filters:
      return self._cluster_device_filters
    self._make_cluster_device_filters()
    return self._cluster_device_filters
  def _make_cluster_device_filters(self):
    """Creates `ClusterDeviceFilters` proto based on the `_device_filters`.
    Raises:
      TypeError: If `_device_filters` is not a dictionary mapping strings to
      a map of task indices and device filters.
    """
    self._cluster_device_filters = device_filters_pb2.ClusterDeviceFilters()
    # Sort by job_name to produce deterministic protobufs.
    for job_name, tasks in sorted(self._device_filters.items()):
      try:
        job_name = compat.as_bytes(job_name)
      except TypeError:
        raise TypeError("Job name %r must be bytes or unicode" % job_name)
      jdf = self._cluster_device_filters.jobs.add()
      jdf.name = job_name
      for i, task_device_filters in sorted(tasks.items()):
        for tdf in task_device_filters:
          try:
            tdf = compat.as_bytes(tdf)
          except TypeError:
            raise TypeError("Device filter %r must be bytes or unicode" % tdf)
          jdf.tasks[i].device_filters.append(tdf)
