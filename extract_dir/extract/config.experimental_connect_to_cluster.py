@tf_export("config.experimental_connect_to_cluster")
def connect_to_cluster(cluster_spec_or_resolver,
                       job_name="localhost",
                       task_index=0,
                       protocol=None,
                       make_master_device_default=True,
                       cluster_device_filters=None):
  """Connects to the given cluster.
  Will make devices on the cluster available to use. Note that calling this more
  than once will work, but will invalidate any tensor handles on the old remote
  devices.
  If the given local job name is not present in the cluster specification, it
  will be automatically added, using an unused port on the localhost.
  Device filters can be specified to isolate groups of remote tasks to avoid
  undesired accesses between workers. Workers accessing resources or launching
  ops / functions on filtered remote devices will result in errors (unknown
  devices). For any remote task, if no device filter is present, all cluster
  devices will be visible; if any device filter is specified, it can only
  see devices matching at least one filter. Devices on the task itself are
  always visible. Device filters can be particially specified.
  For example, for a cluster set up for parameter server training, the following
  device filters might be specified:
  ```python
  cdf = tf.config.experimental.ClusterDeviceFilters()
  # For any worker, only the devices on PS nodes and itself are visible
  for i in range(num_workers):
    cdf.set_device_filters('worker', i, ['/job:ps'])
  # Similarly for any ps, only the devices on workers and itself are visible
  for i in range(num_ps):
    cdf.set_device_filters('ps', i, ['/job:worker'])
  tf.config.experimental_connect_to_cluster(cluster_def,
                                            cluster_device_filters=cdf)
  ```
  Args:
    cluster_spec_or_resolver: A `ClusterSpec` or `ClusterResolver` describing
      the cluster.
    job_name: The name of the local job.
    task_index: The local task index.
    protocol: The communication protocol, such as `"grpc"`. If unspecified, will
      use the default from `python/platform/remote_utils.py`.
    make_master_device_default: If True and a cluster resolver is passed, will
      automatically enter the master task device scope, which indicates the
      master becomes the default device to run ops. It won't do anything if
      a cluster spec is passed. Will throw an error if the caller is currently
      already in some device scope.
    cluster_device_filters: an instance of
      `tf.train.experimental/ClusterDeviceFilters` that specify device filters
      to the remote tasks in cluster.
  """
  if not context.executing_eagerly():
    raise ValueError(
        "`tf.config.experimental_connect_to_cluster` can only be called in "
        "eager mode."
    )
  protocol = protocol or remote_utils.get_default_communication_protocol()
  if isinstance(cluster_spec_or_resolver, server_lib.ClusterSpec):
    cluster_spec = cluster_spec_or_resolver
  elif isinstance(cluster_spec_or_resolver, cluster_resolver.ClusterResolver):
    if cluster_spec_or_resolver.master() in _LOCAL_MASTERS:
      # Do nothing if the master is local.
      return
    cluster_spec = cluster_spec_or_resolver.cluster_spec()
  else:
    raise ValueError(
        "`cluster_spec_or_resolver` must be a `ClusterSpec` or a "
        "`ClusterResolver`.")
  cluster_def = copy.deepcopy(cluster_spec.as_cluster_def())
  if cluster_device_filters:
    if isinstance(cluster_device_filters, server_lib.ClusterDeviceFilters):
      cluster_device_filters = copy.deepcopy(
          cluster_device_filters._as_cluster_device_filters())  # pylint: disable=protected-access
    else:
      raise ValueError("`cluster_device_filters` must be an instance of "
                       "`tf.train.experimental.ClusterDeviceFilters`.")
  # Check whether the server def has changed. We need to do the check before the
  # local job is added to the cluster.
  is_server_def_changed = False
  current_server_def = context.get_server_def()
  if current_server_def and job_name not in cluster_spec.jobs:
    for i, job in enumerate(current_server_def.cluster.job):
      if job.name == job_name:
        del current_server_def.cluster.job[i]
  if (current_server_def is None or current_server_def.cluster != cluster_def or
      current_server_def.job_name != job_name or
      current_server_def.task_index != task_index):
    is_server_def_changed = True
  # Automatically add local job, if not part of the cluster spec.
  if job_name not in cluster_spec.jobs:
    local_port = pywrap_tfe.TF_PickUnusedPortOrDie()
    job_def = cluster_def.job.add()
    job_def.name = job_name
    # TODO(fishx): Update this to make sure remote worker has valid ip address
    # to connect with local.
    job_def.tasks[0] = "localhost:{}".format(local_port)
  if context.context().coordination_service is None:
    service_type = remote_utils.coordination_service_type(protocol)
    service_leader = ""
    # Maybe enable coordination service for the communication protocol
    # TODO(b/243839559): Fix UPTC + Coordination service crashing
    if isinstance(cluster_spec_or_resolver,
                  tpu_cluster_resolver.TPUClusterResolver):
      is_uptc_sess = ".uptc-worker." in cluster_spec_or_resolver.master()
      service_type = remote_utils.coordination_service_type(
          protocol, is_uptc_sess)
      service_leader = cluster_spec_or_resolver.get_coordination_service_leader(
      )
    if service_type:
      # If `enable_health_check` is true, coordination service agent would
      # do connecting (and tasks would send heartbeat if connection is set up)
      # while creating eager contexts. Enabling health check does not mutate
      # coordination service.
      context.context().configure_coordination_service(
          service_type=service_type,
          service_leader=service_leader,
          enable_health_check=False)
  default_session_config = copy.deepcopy(context.context().config)
  for name in cluster_spec.jobs:
    # assuming any of the non-local job is the worker jobs.
    # should we use cluster_spec_or_resolver.get_job_name() instead when
    # it is available?
    # maybe consolicate this with the 'master' logic below
    if name == job_name:
      continue
    default_session_config.experimental.collective_group_leader = (
        f"/job:{name}/replica:0/task:0"
    )
  logging.info("default session config: %s", default_session_config)
  server_def = ServerDef(
      cluster=cluster_def,
      job_name=job_name,
      task_index=task_index,
      protocol=protocol,
      default_session_config=default_session_config,
      cluster_device_filters=cluster_device_filters,
  )
  if is_server_def_changed:
    context.set_server_def(server_def)
  else:
    context.update_server_def(server_def)
  if make_master_device_default and isinstance(
      cluster_spec_or_resolver,
      cluster_resolver.ClusterResolver) and cluster_spec_or_resolver.master():
    master = cluster_spec_or_resolver.master()
    master_job_name = None
    master_task_id = None
    for job_name in cluster_spec.jobs:
      for task_id in cluster_spec.task_indices(job_name):
        task_address = cluster_spec.task_address(job_name, task_id)
        if master in task_address or task_address in master:
          master_job_name = job_name
          master_task_id = task_id
          break
    if not master_job_name:
      raise ValueError(
          "`make_master_device_default` is set to True but cannot find "
          "master %s in the cluster" % master)
    master_device = "/job:{}/replica:0/task:{}".format(master_job_name,
                                                       master_task_id)
    master_device = device_util.canonicalize(master_device)
    current_device = device_util.current()
    if current_device:
      current_device = device_util.canonicalize(current_device)
    if current_device and current_device != master_device:
      raise ValueError("`connect_to_cluster` is called inside existing device "
                       "scope %s, which is different from the master device "
                       "scope %s to enter. This is not allowed." %
                       (current_device, master_device))
    # TODO(b/138389076): Think of the entering device scope behavior in the
    # failure recovery case when dealing with preemptions.
    if not current_device:
      logging.info("Entering into master device scope: %s", master_device)
      ops.device(master_device).__enter__()
