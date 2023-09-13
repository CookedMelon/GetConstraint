@tf_export("summary.trace_on", v1=[])
def trace_on(graph=True, profiler=False):  # pylint: disable=redefined-outer-name
  """Starts a trace to record computation graphs and profiling information.
  Must be invoked in eager mode.
  When enabled, TensorFlow runtime will collect information that can later be
  exported and consumed by TensorBoard. The trace is activated across the entire
  TensorFlow runtime and affects all threads of execution.
  To stop the trace and export the collected information, use
  `tf.summary.trace_export`. To stop the trace without exporting, use
  `tf.summary.trace_off`.
  Args:
    graph: If True, enables collection of executed graphs. It includes ones from
        tf.function invocation and ones from the legacy graph mode. The default
        is True.
    profiler: If True, enables the advanced profiler. Enabling profiler
        implicitly enables the graph collection. The profiler may incur a high
        memory overhead. The default is False.
  """
  if ops.inside_function():
    logging.warn("Cannot enable trace inside a tf.function.")
    return
  if not context.executing_eagerly():
    logging.warn("Must enable trace in eager mode.")
    return
  global _current_trace_context
  with _current_trace_context_lock:
    if _current_trace_context:
      logging.warn("Trace already enabled")
      return
    if graph and not profiler:
      context.context().enable_graph_collection()
    if profiler:
      context.context().enable_run_metadata()
      _profiler.start()
    _current_trace_context = _TraceContext(graph=graph, profiler=profiler)
