@tf_export("summary.create_file_writer", v1=[])
def create_file_writer_v2(logdir,
                          max_queue=None,
                          flush_millis=None,
                          filename_suffix=None,
                          name=None,
                          experimental_trackable=False):
  """Creates a summary file writer for the given log directory.
  Args:
    logdir: a string specifying the directory in which to write an event file.
    max_queue: the largest number of summaries to keep in a queue; will
     flush once the queue gets bigger than this. Defaults to 10.
    flush_millis: the largest interval between flushes. Defaults to 120,000.
    filename_suffix: optional suffix for the event file name. Defaults to `.v2`.
    name: a name for the op that creates the writer.
    experimental_trackable: a boolean that controls whether the returned writer
      will be a `TrackableResource`, which makes it compatible with SavedModel
      when used as a `tf.Module` property.
  Returns:
    A SummaryWriter object.
  """
  if logdir is None:
    raise ValueError("Argument `logdir` cannot be None")
  inside_function = ops.inside_function()
  with ops.name_scope(name, "create_file_writer") as scope, ops.device("cpu:0"):
    # Run init inside an init_scope() to hoist it out of tf.functions.
    with ops.init_scope():
      if context.executing_eagerly():
        _check_create_file_writer_args(
            inside_function,
            logdir=logdir,
            max_queue=max_queue,
            flush_millis=flush_millis,
            filename_suffix=filename_suffix)
      logdir = ops.convert_to_tensor(logdir, dtype=dtypes.string)
      if max_queue is None:
        max_queue = constant_op.constant(10)
      if flush_millis is None:
        flush_millis = constant_op.constant(2 * 60 * 1000)
      if filename_suffix is None:
        filename_suffix = constant_op.constant(".v2")
      def create_fn():
        # Use unique shared_name to prevent resource sharing in eager mode, but
        # otherwise use a fixed shared_name to allow SavedModel TF 1.x loading.
        if context.executing_eagerly():
          shared_name = context.anonymous_name()
        else:
          shared_name = ops.name_from_scope_name(scope)  # pylint: disable=protected-access
        return gen_summary_ops.summary_writer(
            shared_name=shared_name, name=name)
      init_op_fn = functools.partial(
          gen_summary_ops.create_summary_file_writer,
          logdir=logdir,
          max_queue=max_queue,
          flush_millis=flush_millis,
          filename_suffix=filename_suffix)
      if experimental_trackable:
        return _TrackableResourceSummaryWriter(
            create_fn=create_fn, init_op_fn=init_op_fn)
      else:
        return _ResourceSummaryWriter(
            create_fn=create_fn, init_op_fn=init_op_fn)
