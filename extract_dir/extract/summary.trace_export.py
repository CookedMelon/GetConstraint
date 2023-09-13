@tf_export("summary.trace_export", v1=[])
def trace_export(name, step=None, profiler_outdir=None):
  """Stops and exports the active trace as a Summary and/or profile file.
  Stops the trace and exports all metadata collected during the trace to the
  default SummaryWriter, if one has been set.
  Args:
    name: A name for the summary to be written.
    step: Explicit `int64`-castable monotonic step value for this summary. If
      omitted, this defaults to `tf.summary.experimental.get_step()`, which must
      not be None.
    profiler_outdir: Output directory for profiler. It is required when profiler
      is enabled when trace was started. Otherwise, it is ignored.
  Raises:
    ValueError: if a default writer exists, but no step was provided and
      `tf.summary.experimental.get_step()` is None.
  """
  # TODO(stephanlee): See if we can remove profiler_outdir and infer it from
  # the SummaryWriter's logdir.
  global _current_trace_context
  if ops.inside_function():
    logging.warn("Cannot export trace inside a tf.function.")
    return
  if not context.executing_eagerly():
    logging.warn("Can only export trace while executing eagerly.")
    return
  with _current_trace_context_lock:
    if _current_trace_context is None:
      raise ValueError("Must enable trace before export through "
                       "tf.summary.trace_on.")
    graph, profiler = _current_trace_context  # pylint: disable=redefined-outer-name
    if profiler and profiler_outdir is None:
      raise ValueError("Argument `profiler_outdir` is not specified.")
  run_meta = context.context().export_run_metadata()
  if graph and not profiler:
    run_metadata_graphs(name, run_meta, step)
  else:
    run_metadata(name, run_meta, step)
  if profiler:
    _profiler.save(profiler_outdir, _profiler.stop())
  trace_off()
