@tf_export("summary.graph", v1=[])
def graph(graph_data):
  """Writes a TensorFlow graph summary.
  Write an instance of `tf.Graph` or `tf.compat.v1.GraphDef` as summary only
  in an eager mode. Please prefer to use the trace APIs (`tf.summary.trace_on`,
  `tf.summary.trace_off`, and `tf.summary.trace_export`) when using
  `tf.function` which can automatically collect and record graphs from
  executions.
  Usage Example:
  ```py
  writer = tf.summary.create_file_writer("/tmp/mylogs")
  @tf.function
  def f():
    x = constant_op.constant(2)
    y = constant_op.constant(3)
    return x**y
  with writer.as_default():
    tf.summary.graph(f.get_concrete_function().graph)
  # Another example: in a very rare use case, when you are dealing with a TF v1
  # graph.
  graph = tf.Graph()
  with graph.as_default():
    c = tf.constant(30.0)
  with writer.as_default():
    tf.summary.graph(graph)
  ```
  Args:
    graph_data: The TensorFlow graph to write, as a `tf.Graph` or a
      `tf.compat.v1.GraphDef`.
  Returns:
    True on success, or False if no summary was written because no default
    summary writer was available.
  Raises:
    ValueError: `graph` summary API is invoked in a graph mode.
  """
  if not context.executing_eagerly():
    raise ValueError("graph() cannot be invoked inside a graph context.")
  writer = _summary_state.writer
  if writer is None:
    return constant_op.constant(False)
  with ops.device("cpu:0"):
    if not should_record_summaries():
      return constant_op.constant(False)
    if isinstance(graph_data, (ops.Graph, graph_pb2.GraphDef)):
      tensor = ops.convert_to_tensor(
          _serialize_graph(graph_data), dtypes.string)
    else:
      raise ValueError("Argument 'graph_data' is not tf.Graph or "
                       "tf.compat.v1.GraphDef. Received graph_data="
                       f"{graph_data} of type {type(graph_data).__name__}.")
    gen_summary_ops.write_graph_summary(
        writer._resource,  # pylint: disable=protected-access
        # Graph does not have step. Set to 0.
        0,
        tensor,
    )
    return constant_op.constant(True)
