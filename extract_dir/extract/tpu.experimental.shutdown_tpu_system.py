@tf_export("tpu.experimental.shutdown_tpu_system")
def shutdown_tpu_system(cluster_resolver=None):
  """Shuts down the TPU devices.
  This will clear all caches, even those that are maintained through sequential
  calls to tf.tpu.experimental.initialize_tpu_system, such as the compilation
  cache.
  Args:
    cluster_resolver: A tf.distribute.cluster_resolver.TPUClusterResolver,
        which provides information about the TPU cluster.
  Raises:
    RuntimeError: If no TPU devices found for eager execution or if run in a
        tf.function.
  """
  job = None
  if cluster_resolver is None:
    # If no cluster resolver is specified, and running eagerly, execute the init
    # ops in the current device scope.
    if context.executing_eagerly():
      curr_device = device.DeviceSpec.from_string(context.context().device_name)
      if curr_device.job is not None:
        job = "{}/replica:0/task:0".format(curr_device.job)
    cluster_resolver = TPUClusterResolver("")
  assert isinstance(cluster_resolver, TPUClusterResolver)
  tpu_name = compat.as_text(cluster_resolver._tpu)  # pylint: disable=protected-access
  if tpu_name not in _INITIALIZED_TPU_SYSTEMS:
    logging.warning("You are shutting down a TPU system %s that has not been "
                    "initialized." % tpu_name)
  logging.info("Shutting down the TPU system: %s", tpu_name)
  if context.executing_eagerly():
    # This function looks as it is for the following non-intuitive reasons.
    # tpu.shutdown_system creates a dummy op whose sole purpose is to trigger
    # DistributedTPURewritePass. This pass actually adds real ops that
    # shutdown the TPU system. Thus, we can't simply run tpu.shutdown_system
    # eagerly. We need to wrap it in defun and trigger the rewrite passes on it.
    if tpu_name not in _LOCAL_MASTERS:
      # Explicitly place the tpu.shutdown_system in the first worker to
      # avoid the output node match multiple devices error.
      job = "{}/replica:0/task:0".format(cluster_resolver.get_job_name())
    @function(autograph=False)
    def _tpu_shutdown_fn():
      tpu.shutdown_system(job=job)
    # The TPU_SYSTEM device must match the device used in tpu.shutdown_system
    # exactly, otherwise you can get errors if there are multiple TPU_SYSTEM
    # devices available.
    run_eagerly = functions_run_eagerly()
    if run_eagerly:
      logging.warning(
          "It looks like tf.function behavior was disabled, perhaps using"
          " tf.config.run_functions_eagerly."
          " tf.tpu.experimental.shutdown_tpu_system requires tf.function to"
          " work. This primitive will override the disable."
      )
      run_functions_eagerly(False)
    try:
      with ops.device(tpu._tpu_system_device_name(job)):  # pylint: disable=protected-access
        _tpu_shutdown_fn()
    finally:
      if run_eagerly is not None:
        run_functions_eagerly(run_eagerly)
    # Clear out the eager context caches since the memory is invalid now.
    logging.info("Clearing out eager caches")
    context.context()._clear_caches()  # pylint: disable=protected-access
    context.context().clear_kernel_cache()
  elif not ops.executing_eagerly_outside_functions():
    master = cluster_resolver.master()
    cluster_spec = cluster_resolver.cluster_spec()
    session_config = config_pb2.ConfigProto(allow_soft_placement=True)
    if cluster_spec:
      session_config.cluster_def.CopyFrom(cluster_spec.as_cluster_def())
    with ops.Graph().as_default():
      with session_lib.Session(config=session_config, target=master) as sess:
        sess.run(tpu.shutdown_system())
  else:
    raise RuntimeError(
        "initialize_tpu_system is not supported within "
        "tf.functions.  You should call initialize_tpu_system outside of your tf.function. "
    )
  logging.info("Finished shutting down TPU system.")
  if tpu_name in _INITIALIZED_TPU_SYSTEMS:
    del _INITIALIZED_TPU_SYSTEMS[tpu_name]
